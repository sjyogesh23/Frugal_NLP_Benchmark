"""
MCP server exposing the frugal NLP pipeline: create, test and reference models by use case.

    python mcp_server.py           serve over stdio for any MCP client
    python mcp_server.py --demo    spawn this server and drive it with a Python MCP client

Every tool is a thin wrapper around frugal_nlp, so an agent gets the same reproducible protocol as the CLI.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import frugal_nlp as fn

try:
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:
    from mcp.server.fastmcp import FastMCP as _Server

mcp = _Server("frugal-nlp")


def _records(df):
    return json.loads(df.to_json(orient="records"))


@mcp.tool()
def list_use_cases() -> list[dict]:
    """List available use cases (built-in HF datasets and registered private CSVs) with their labels."""
    return fn.use_cases()


@mcp.tool()
def register_use_case(name: str, csv_path: str, text_col: str, label_col: str, description: str = "") -> dict:
    """Add a private labelled CSV as a new use case. The data is encrypted at rest and never sent out."""
    return fn.register_use_case(name, csv_path, text_col, label_col, description)


@mcp.tool()
def create_model(use_case: str, n_per_class: int = 32, base_model: str = fn.DEFAULT_BASE,
                 epochs: int = 20, lora_r: int = 8, seed: int = 42, n_test: int = 300) -> dict:
    """Fine-tune a small model with LoRA on `n_per_class` labelled examples per class, register it in the
    SQLite registry with its training energy (CodeCarbon), then evaluate it on the fixed test sample."""
    cfg = fn.TrainConfig(use_case=use_case, n_per_class=n_per_class, base_model=base_model,
                         epochs=epochs, lora_r=lora_r, seed=seed)
    return fn.create_model(cfg, n_test=n_test)


@mcp.tool()
def test_model(model_id: str, n_test: int = 300) -> dict:
    """Evaluate a registered model: accuracy, macro-F1, latency, Wh per 1k predictions, CO2."""
    return fn.test_model(model_id, n_test)


@mcp.tool()
def run_llm_baseline(use_case: str, provider: str = "local", model_name: str | None = None,
                     n_test: int = 300) -> dict:
    """Zero-shot LLM baseline. provider='local' (Qwen2.5-0.5B on this machine) or 'gemini' (API;
    texts are PII-redacted before leaving the machine)."""
    return fn.run_llm_baseline(use_case, provider, model_name, n_test)


@mcp.tool()
def predict(model_id: str, texts: list[str]) -> list[dict]:
    """Classify texts with a registered model."""
    return fn.predict(model_id, texts)


@mcp.tool()
def list_models(use_case: str | None = None) -> list[dict]:
    """List registered models (optionally for one use case)."""
    return _records(fn.list_models(use_case))


@mcp.tool()
def leaderboard(use_case: str | None = None) -> list[dict]:
    """Latest evaluation per model ranked by macro-F1, with energy and privacy columns."""
    return _records(fn.leaderboard(use_case))


@mcp.tool()
def recommend_model(use_case: str, max_wh_per_1k: float | None = None, min_f1: float = 0.0,
                    f1_tolerance: float = 0.02, local_only: bool = False) -> dict:
    """Frugal reference model for a use case: the lowest-energy model within `f1_tolerance` macro-F1 of the
    best one. Optional energy budget (Wh per 1k predictions), quality floor and on-premises-only filter."""
    return fn.recommend(use_case, max_wh_per_1k, min_f1, f1_tolerance, local_only)


@mcp.tool()
def energy_breakeven(use_case: str) -> list[dict]:
    """Number of predictions after which each small model (training included) uses less energy than each LLM."""
    return _records(fn.breakeven(use_case))


@mcp.tool()
def redact_pii(text: str) -> dict:
    """Show how a text is anonymised before any remote LLM call."""
    clean, n = fn.redact(text)
    return {"text": clean, "n_redacted": n}


@mcp.resource("registry://leaderboard")
def leaderboard_resource() -> str:
    """Leaderboard as markdown."""
    return fn.leaderboard().to_markdown(index=False)


def _maybe_json(text: str):
    try:
        return json.loads(text) if text.strip()[:1] in ("[", "{") else text
    except json.JSONDecodeError:
        return text


async def _demo():
    """Minimal MCP client: start this file as a stdio server, list tools, run a small pipeline."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=sys.executable, args=[str(Path(__file__).resolve())],
                                   cwd=str(Path(__file__).parent))
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        async def call(name, **kw):
            r = await session.call_tool(name, kw)
            sc = getattr(r, "structured_content", None) or getattr(r, "structuredContent", None)
            if isinstance(sc, dict):
                data = sc.get("result", sc)
            else:
                texts = [getattr(c, "text", "") for c in r.content]
                data = [_maybe_json(t) for t in texts]
                data = data[0] if len(data) == 1 and isinstance(data[0], dict) else data
            print(f"\n>>> {name}({', '.join(f'{k}={v!r}' for k, v in kw.items())})\n"
                  f"{json.dumps(data, indent=1, default=str)[:1500]}")
            return data

        tools = await session.list_tools()
        print("tools:", [t.name for t in tools.tools])
        await call("list_use_cases")
        await call("redact_pii", text="Call me on +44 7700 900123 or mail jane.doe@corp.com")
        res = await call("create_model", use_case="spam", n_per_class=16, epochs=15, n_test=200)
        mid = res["model_id"] if isinstance(res, dict) else None
        if mid:
            await call("predict", model_id=mid, texts=["WINNER!! Claim your free prize now, text WIN to 80086",
                                                       "Are we still on for lunch tomorrow?"])
        await call("leaderboard", use_case="spam")
        await call("recommend_model", use_case="spam")
        await call("recommend_model", use_case="spam", local_only=True)


if __name__ == "__main__":
    if "--demo" in sys.argv:
        asyncio.run(_demo())
    else:
        mcp.run()
