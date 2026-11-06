"""
One command for everything:

    python run.py              web UI + API and Jupyter
    python run.py --only ui    web UI only
    python run.py --open       also open the UI in the browser

Prints the URL of every service; Ctrl+C stops them all.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

try:
    import uvicorn
    from starlette.applications import Starlette
    from starlette.concurrency import run_in_threadpool
    from starlette.requests import Request
    from starlette.responses import FileResponse, JSONResponse
    from starlette.routing import Route

    import frugal_nlp as fn
except ImportError as e:
    sys.exit(f"Missing package ({e.name}). Activate the project venv first:\n"
             f"  .venv\\Scripts\\activate      (Windows)\n  source .venv/bin/activate   (Linux/macOS)\n"
             f"then run: python run.py")

ROOT = Path(__file__).resolve().parent
PY = sys.executable
GPU_LOCK = threading.Lock()
BUSY = {"job": None, "since": None}
SERVICES: dict[str, str] = {}
_RESERVED: set[int] = set()


def free_port(preferred: int, host: str) -> int:
    """First port >= preferred that can really be bound and was not handed to another service."""
    for port in range(preferred, preferred + 100):
        if port in _RESERVED:
            continue
        with socket.socket() as s:
            try:
                s.bind((host, port))
            except OSError:
                continue
        _RESERVED.add(port)
        return port
    raise RuntimeError(f"no free port near {preferred}")


def already_running(host: str) -> dict | None:
    """Services of a run.py that is already up, so a second start just prints them."""
    try:
        with urllib.request.urlopen(f"http://{host}:7860/api/status", timeout=1.5) as r:
            return json.load(r).get("services")
    except Exception:
        return None


def kill_children_on_exit() -> None:
    """Windows: put run.py in a job object so Jupyter dies with it, even if the window is closed."""
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", ctypes.c_ulonglong * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    job = k32.CreateJobObjectW(None, None)
    info = Extended()
    info.BasicLimitInformation.LimitFlags = 0x2000
    k32.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info))
    k32.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(k32.GetCurrentProcess()))
    kill_children_on_exit.handle = job


def records(df):
    return json.loads(df.to_json(orient="records"))


def heavy(name, func, *args, **kwargs):
    """Run a GPU job under the lock and expose it in /api/status."""
    with GPU_LOCK:
        BUSY.update(job=name, since=time.time())
        try:
            return func(*args, **kwargs)
        finally:
            BUSY.update(job=None, since=None)


async def api(request: Request, handler) -> JSONResponse:
    try:
        body = await request.json() if request.method == "POST" else {}
        return JSONResponse(await run_in_threadpool(handler, request.query_params, body))
    except (KeyError, ValueError) as e:
        return JSONResponse({"error": str(e).strip("'\"")}, status_code=400)
    except Exception as e:
        fn.log.exception("API error")
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


def h_status(q, b):
    models = fn.list_models()
    return {"device": fn.DEVICE, "gpu": fn.env_snapshot()["gpu"], "n_models": len(models),
            "n_evals": int(fn.query("SELECT COUNT(*) AS n FROM evaluations").n[0]),
            "gemini": bool(os.getenv("GEMINI_API_KEY")), "default_base": fn.DEFAULT_BASE,
            "busy": BUSY["job"], "busy_for": round(time.time() - BUSY["since"], 1) if BUSY["since"] else None,
            "services": SERVICES}


def h_overview(q, b):
    uc = q.get("use_case")
    lb = fn.leaderboard(uc)
    return {"leaderboard": records(lb), "recommend": fn.recommend(uc) if uc else None,
            "local_recommend": fn.recommend(uc, local_only=True) if uc else None,
            "breakeven": records(fn.breakeven(uc)) if uc else []}


def h_train(q, b):
    cfg = fn.TrainConfig(use_case=b["use_case"], base_model=b.get("base_model") or fn.DEFAULT_BASE,
                         n_per_class=int(b.get("n_per_class", 32)), epochs=int(b.get("epochs", 20)),
                         lora_r=int(b.get("lora_r", 8)), seed=int(b.get("seed", 42)))
    return heavy(f"Training {cfg.use_case}", fn.create_model, cfg, n_test=int(b.get("n_test", 300)))


def h_llm(q, b):
    return heavy(f"{b.get('provider', 'local')} baseline on {b['use_case']}", fn.run_llm_baseline,
                 b["use_case"], b.get("provider", "local"), None, int(b.get("n_test", 300)))


def h_predict(q, b):
    texts = [t for t in b.get("texts", []) if t.strip()][:200]
    if not texts:
        raise ValueError("enter at least one text")
    kind = fn.get_model(b["model_id"])["kind"]
    t0 = time.perf_counter()
    out = heavy("Predicting", fn.predict, b["model_id"], texts) if kind != "slm" else fn.predict(b["model_id"], texts)
    return {"predictions": out, "ms": round((time.perf_counter() - t0) * 1000, 1)}


def h_redact(q, b):
    text, n = fn.redact(b.get("text", ""))
    return {"text": text, "n": n}


def h_register(q, b):
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".csv")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(b["csv"])
        return fn.register_use_case(b["name"], path, b["text_col"], b["label_col"], b.get("description", ""))
    finally:
        os.unlink(path)


ROUTES = {
    "status": ("GET", h_status), "use-cases": ("GET", lambda q, b: fn.use_cases()),
    "overview": ("GET", h_overview), "models": ("GET", lambda q, b: records(fn.list_models(q.get("use_case")))),
    "train": ("POST", h_train), "llm": ("POST", h_llm), "predict": ("POST", h_predict),
    "redact": ("POST", h_redact), "register": ("POST", h_register),
}


def build_app() -> Starlette:
    def endpoint(handler):
        async def ep(request):
            return await api(request, handler)
        return ep
    routes = [Route("/", lambda r: FileResponse(ROOT / "ui.html"))]
    for name, (method, handler) in ROUTES.items():
        routes.append(Route(f"/api/{name}", endpoint(handler), methods=[method]))
    return Starlette(routes=routes)


class Server(uvicorn.Server):
    """uvicorn server that leaves Ctrl+C to run.py, so all servers stop together."""
    @contextlib.contextmanager
    def capture_signals(self):
        yield


def server(app, host, port) -> Server:
    return Server(uvicorn.Config(app, host=host, port=port, log_level="warning", lifespan="on"))


def start_subprocess(cmd) -> subprocess.Popen:
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    return subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, creationflags=flags, start_new_session=os.name != "nt")


async def main(a) -> None:
    want = set(a.only.split(","))
    servers, procs = [], []

    if "ui" in want:
        port = free_port(7860, a.host)
        servers.append(server(build_app(), a.host, port))
        SERVICES["Web UI"] = f"http://{a.host}:{port}/"
        SERVICES["REST API"] = f"http://{a.host}:{port}/api/status"
    if "jupyter" in want:
        port, token = free_port(7861, a.host), secrets.token_hex(12)
        procs.append(start_subprocess([PY, "-m", "notebook", "--no-browser", f"--port={port}", f"--ip={a.host}",
                                       f"--IdentityProvider.token={token}", f"--ServerApp.root_dir={ROOT}"]))
        SERVICES["Jupyter notebook"] = f"http://{a.host}:{port}/notebooks/analysis.ipynb?token={token}"

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, getattr(signal, "SIGBREAK", None), getattr(signal, "SIGTERM", None)):
        if sig is not None:
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop.set))

    tasks = [asyncio.create_task(s.serve()) for s in servers]
    await asyncio.sleep(1.5)
    failed = [t for t in tasks if t.done()]
    if failed:
        print("\n  A server failed to start (see the error above). Stopping.\n", flush=True)
        stop.set()
    else:
        width = max(map(len, SERVICES)) + 2
        print("\n  Frugal NLP Benchmark is running\n", flush=True)
        for name, url in SERVICES.items():
            print(f"  {name:<{width}} {url}", flush=True)
        if "jupyter" in want:
            print("\n  (Jupyter needs a few seconds before its page loads)", flush=True)
        print(f"\n  MCP over stdio (any MCP client): {PY} mcp_server.py\n  Press Ctrl+C to stop everything\n", flush=True)
        if a.open and "Web UI" in SERVICES:
            threading.Timer(1.0, webbrowser.open, [SERVICES["Web UI"]]).start()

    waiter = asyncio.create_task(stop.wait())
    await asyncio.wait([waiter, *tasks], return_when=asyncio.FIRST_COMPLETED)

    print("  Stopping...", flush=True)
    for s in servers:
        s.should_exit = True
    await asyncio.wait(tasks, timeout=5)
    for p in procs:
        p.terminate()
    for p in procs:
        with contextlib.suppress(subprocess.TimeoutExpired):
            p.wait(timeout=5)
    waiter.cancel()
    print("  Stopped.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Start the Frugal NLP services")
    ap.add_argument("--only", default="ui,jupyter", help="comma list of ui,jupyter")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true", help="open the web UI in the browser")
    args = ap.parse_args()

    running = already_running(args.host)
    if running:
        print("\n  Frugal NLP Benchmark is already running:\n")
        for name, url in running.items():
            print(f"  {name:<24} {url}")
        print("\n  Stop that window with Ctrl+C before starting a new one.\n")
        sys.exit(0)

    kill_children_on_exit()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main(args))
