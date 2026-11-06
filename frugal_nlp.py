"""
Frugal NLP Benchmark — few-shot LoRA small models vs LLMs.

One module holds the whole pipeline so the CLI, the MCP server (mcp_server.py)
and the dashboard (app.py) share exactly the same code:

    data      -> HF datasets or private CSVs (encrypted at rest with Fernet)
    privacy   -> regex PII redaction before any text leaves the machine
    training  -> PyTorch loop + PEFT LoRA on a small encoder, few labels per class
    energy    -> CodeCarbon around every training / inference run
    registry  -> SQLite (use cases, models, evaluations, leaderboard SQL)
    LLM       -> Gemini API (zero-shot, batched) and a local small instruct LLM
    protocol  -> fixed seeds, fixed test sample, config + data hashes, env snapshot

Run `python frugal_nlp.py -h` for the CLI.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import platform
import random
import re
import sqlite3
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
load_dotenv(ENV_PATH)

DB_PATH = Path(os.getenv("FRUGAL_DB", ROOT / "registry.db"))
MODELS_DIR, DATA_DIR, RESULTS_DIR = ROOT / "models", ROOT / "data", ROOT / "results"
for _d in (MODELS_DIR, DATA_DIR, RESULTS_DIR):
    _d.mkdir(exist_ok=True)

DEVICE = os.getenv("FRUGAL_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
COUNTRY = os.getenv("CODECARBON_COUNTRY", "IND")
DEFAULT_BASE = "BAAI/bge-small-en-v1.5"
DEFAULT_LOCAL_LLM = "Qwen/Qwen2.5-0.5B-Instruct"
TEST_SEED = 1234
GEMINI_WH_PER_PROMPT = float(os.getenv("GEMINI_WH_PER_PROMPT", "0.24"))
GEMINI_G_CO2_PER_PROMPT = float(os.getenv("GEMINI_G_CO2_PER_PROMPT", "0.03"))

logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("frugal")
log.setLevel(logging.INFO)
for _noisy in ("codecarbon", "httpx", "transformers", "datasets", "google_genai", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("CODECARBON_LOG_LEVEL", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

TASKS = {
    "sentiment": dict(hf="stanfordnlp/sst2", text="sentence", label="label", test_split="validation",
                      labels=["negative", "positive"], description="Movie review sentiment (SST-2)"),
    "topic": dict(hf="fancyzhx/ag_news", text="text", label="label", test_split="test",
                  labels=["World", "Sports", "Business", "Sci/Tech"], description="News topic classification (AG News)"),
    "spam": dict(hf="ucirvine/sms_spam", text="sms", label="label", test_split=None,
                 labels=["ham", "spam"], description="SMS spam detection, private-message scenario"),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def env_snapshot() -> dict:
    import peft
    import transformers
    return {
        "python": platform.python_version(), "os": platform.platform(), "torch": torch.__version__,
        "transformers": transformers.__version__, "peft": peft.__version__, "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
    }


def _luhn(digits: str) -> bool:
    total = 0
    for i, d in enumerate(map(int, reversed(digits))):
        total += d if i % 2 == 0 else (d * 2 - 9 if d > 4 else d * 2)
    return total % 10 == 0


def _is_card(m: re.Match) -> bool:
    digits = re.sub(r"\D", "", m.group())
    return 13 <= len(digits) <= 16 and _luhn(digits)


def _is_phone(m: re.Match) -> bool:
    s = m.group()
    digits = re.sub(r"\D", "", s)
    if re.fullmatch(r"\d{4}[-./]\d{1,2}[-./]\d{1,2}|\d{1,2}[-./]\d{1,2}[-./]\d{2,4}", s):
        return False
    if re.fullmatch(r"(?:19|20)\d\d(?:\s+(?:19|20)\d\d)+", s):
        return False
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", s):
        return False
    return 9 <= len(digits) <= 15 and (s.startswith("+") or not re.fullmatch(r"\d+", s) or len(digits) >= 10)


PII_PATTERNS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), None),
    ("URL", re.compile(r"(?:https?://|www\.)\S+", re.I), None),
    ("IP", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"), None),
    ("CARD", re.compile(r"\b\d(?:[ -]?\d){12,15}\b"), _is_card),
    ("PHONE", re.compile(r"(?<![\w#])(?:\+|\()?\d[\d ().-]{7,}\d\b"), _is_phone),
]


def redact(text: str) -> tuple[str, int]:
    """Replace e-mails, URLs, IPs, card and phone numbers with [TAG]. Returns (text, n_replacements)."""
    total = 0
    for tag, pattern, valid in PII_PATTERNS:
        def sub(m, tag=tag, valid=valid):
            nonlocal total
            if valid is None or valid(m):
                total += 1
                return f"[{tag}]"
            return m.group()
        text = pattern.sub(sub, text)
    return text, total


def _fernet():
    from cryptography.fernet import Fernet
    from dotenv import dotenv_values
    key = (os.getenv("DATA_ENCRYPTION_KEY") or dotenv_values(ENV_PATH).get("DATA_ENCRYPTION_KEY") or "").strip()
    if not key:
        key = Fernet.generate_key().decode()
        lines = ENV_PATH.read_text().splitlines() if ENV_PATH.exists() else []
        is_key = lambda l: re.match(r"\s*DATA_ENCRYPTION_KEY\s*=", l)
        if any(is_key(l) for l in lines):
            lines = [f"DATA_ENCRYPTION_KEY={key}" if is_key(l) else l for l in lines]
        else:
            lines.append(f"DATA_ENCRYPTION_KEY={key}")
        ENV_PATH.write_text("\n".join(lines) + "\n")
        os.environ["DATA_ENCRYPTION_KEY"] = key
        log.info("generated DATA_ENCRYPTION_KEY in .env")
    return Fernet(key.encode())


def save_encrypted(df: pd.DataFrame, path: Path) -> None:
    path.write_bytes(_fernet().encrypt(df.to_json(orient="records").encode()))


def load_encrypted(path: Path) -> pd.DataFrame:
    return pd.read_json(io.StringIO(_fernet().decrypt(path.read_bytes()).decode()), orient="records",
                        dtype={"text": str, "label": int})


SCHEMA = """
CREATE TABLE IF NOT EXISTS use_cases(
    name TEXT PRIMARY KEY, description TEXT, source TEXT, labels TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS models(
    model_id TEXT PRIMARY KEY, use_case TEXT, kind TEXT, base_model TEXT,
    n_per_class INTEGER, n_train INTEGER, trainable_params INTEGER, total_params INTEGER,
    path TEXT, labels TEXT, config TEXT, config_hash TEXT, data_hash TEXT,
    train_seconds REAL, train_kwh REAL, train_co2_kg REAL, created_at TEXT);
CREATE TABLE IF NOT EXISTS evaluations(
    eval_id INTEGER PRIMARY KEY AUTOINCREMENT, model_id TEXT REFERENCES models(model_id),
    use_case TEXT, n_test INTEGER, test_hash TEXT, accuracy REAL, macro_f1 REAL,
    seconds REAL, kwh REAL, co2_kg REAL, wh_per_1k REAL, latency_ms REAL, energy_source TEXT,
    data_left_premises INTEGER, pii_redacted INTEGER, created_at TEXT);
CREATE INDEX IF NOT EXISTS idx_eval_model ON evaluations(model_id, use_case);
"""

LEADERBOARD_SQL = """
WITH latest AS (
    SELECT * FROM evaluations
    WHERE eval_id IN (SELECT MAX(eval_id) FROM evaluations GROUP BY model_id, use_case))
SELECT l.use_case, m.model_id, m.kind, m.base_model, m.n_train,
       ROUND(l.macro_f1, 4) AS macro_f1, ROUND(l.accuracy, 4) AS accuracy,
       ROUND(l.wh_per_1k, 4) AS wh_per_1k, ROUND(l.latency_ms, 2) AS latency_ms,
       ROUND(COALESCE(m.train_kwh, 0) * 1000, 4) AS train_wh,
       ROUND(l.co2_kg * 1000, 5) AS eval_g_co2, l.energy_source, l.data_left_premises,
       m.trainable_params, l.n_test,
       RANK() OVER (PARTITION BY l.use_case ORDER BY l.macro_f1 DESC) AS rank_f1
FROM latest l JOIN models m USING (model_id)
WHERE (:use_case IS NULL OR l.use_case = :use_case)
ORDER BY l.use_case, rank_f1
"""


@contextmanager
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def _upsert(table: str, row: dict) -> None:
    cols = ", ".join(row)
    with db() as con:
        con.execute(f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({', '.join('?' * len(row))})",
                    [json.dumps(v) if isinstance(v, (dict, list)) else v for v in row.values()])


def query(sql: str, params: dict | tuple = ()) -> pd.DataFrame:
    with db() as con:
        return pd.read_sql_query(sql, con, params=params)


def get_model(model_id: str) -> dict:
    with db() as con:
        row = con.execute("SELECT * FROM models WHERE model_id = ?", (model_id,)).fetchone()
    if row is None:
        raise KeyError(f"unknown model_id {model_id!r}; see list_models()")
    out = dict(row)
    out["labels"] = json.loads(out["labels"])
    out["config"] = json.loads(out["config"] or "{}")
    return out


def list_models(use_case: str | None = None) -> pd.DataFrame:
    return query("""SELECT model_id, use_case, kind, base_model, n_per_class, n_train, trainable_params,
                           ROUND(train_kwh*1000, 4) AS train_wh, train_seconds, created_at
                    FROM models WHERE (:u IS NULL OR use_case = :u) ORDER BY use_case, created_at""",
                 {"u": use_case})


def leaderboard(use_case: str | None = None) -> pd.DataFrame:
    return query(LEADERBOARD_SQL, {"use_case": use_case})


def recommend(use_case: str, max_wh_per_1k: float | None = None, min_f1: float = 0.0,
              f1_tolerance: float = 0.02, local_only: bool = False) -> dict:
    """Frugal reference model for a use case: among models within `f1_tolerance` of the best macro-F1
    (after the energy / quality / privacy constraints), pick the one with the lowest Wh per 1k predictions."""
    lb = leaderboard(use_case)
    if max_wh_per_1k is not None:
        lb = lb[lb.wh_per_1k <= max_wh_per_1k]
    if local_only:
        lb = lb[lb.data_left_premises == 0]
    lb = lb[(lb.macro_f1 >= min_f1) & (lb.energy_source != "unavailable")]
    if lb.empty:
        return {"use_case": use_case, "recommendation": None, "reason": "no evaluated model meets the constraints"}
    top = lb.macro_f1.max()
    best = lb[lb.macro_f1 >= top - f1_tolerance].sort_values(["wh_per_1k", "macro_f1"], ascending=[True, False]).iloc[0]
    reason = (f"lowest energy among models within {f1_tolerance} macro-F1 of the best ({top:.4f})"
              if best.macro_f1 < top else "best macro-F1")
    return {"use_case": use_case, "recommendation": best.model_id, "reason": reason,
            **json.loads(best.drop(["model_id", "use_case"]).to_json())}


def use_cases() -> list[dict]:
    custom = query("SELECT name, description, labels, source FROM use_cases").to_dict("records")
    builtin = [{"name": k, "description": v["description"], "labels": v["labels"], "source": v["hf"]}
               for k, v in TASKS.items()]
    return builtin + [{**c, "labels": json.loads(c["labels"])} for c in custom]


def register_use_case(name: str, csv_path: str, text_col: str, label_col: str, description: str = "") -> dict:
    """Ingest a private CSV as a new use case. The raw file is stored encrypted, never in clear text."""
    if name in TASKS:
        raise ValueError(f"{name!r} is a built-in use case")
    if not re.fullmatch(r"[\w-]{1,64}", name):
        raise ValueError("use-case name must be 1-64 letters, digits, '_' or '-'")
    df = pd.read_csv(csv_path, dtype={label_col: str})[[text_col, label_col]].dropna()
    labels = df[label_col].unique().tolist()
    numeric = all(l.lstrip("-").isdigit() for l in labels)
    labels = sorted(labels, key=int) if numeric else sorted(labels)
    df = pd.DataFrame({"text": df[text_col].astype(str), "label": df[label_col].astype(str).map(labels.index)})
    path = DATA_DIR / f"{name}.enc"
    save_encrypted(df, path)
    _upsert("use_cases", {"name": name, "description": description or f"custom: {Path(csv_path).name}",
                          "source": str(path), "labels": labels, "created_at": now()})
    return {"name": name, "n_rows": len(df), "labels": labels, "stored_encrypted": str(path)}


def _frames(use_case: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str]:
    """Full (train_pool, test_pool, labels, description). Pools never overlap."""
    if use_case in TASKS:
        from datasets import load_dataset
        t = TASKS[use_case]
        ds = load_dataset(t["hf"])
        rename = {t["text"]: "text", t["label"]: "label"}
        pool = ds["train"].to_pandas().rename(columns=rename)[["text", "label"]]
        if t["test_split"]:
            test = ds[t["test_split"]].to_pandas().rename(columns=rename)[["text", "label"]]
        else:
            test = pool.sample(frac=0.2, random_state=0)
            pool = pool.drop(test.index)
        return pool, test, t["labels"], t["description"]
    with db() as con:
        row = con.execute("SELECT * FROM use_cases WHERE name = ?", (use_case,)).fetchone()
    if row is None:
        raise KeyError(f"unknown use case {use_case!r}; built-in: {list(TASKS)}")
    df = load_encrypted(Path(row["source"]))
    test = df.groupby("label", group_keys=False).sample(frac=0.3, random_state=0)
    return df.drop(test.index), test, json.loads(row["labels"]), row["description"]


def load_task(use_case: str, n_per_class: int = 32, n_test: int = 300, seed: int = 42):
    """Few-shot train sample (n_per_class per label) + a fixed test sample shared by every system."""
    pool, test, labels, desc = _frames(use_case)
    pool = pool[pool.label >= 0]
    parts = [g.sample(min(n_per_class, len(g)), random_state=seed) for _, g in pool.groupby("label")]
    train = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    test = test[test.label >= 0]
    test = test.sample(min(n_test, len(test)), random_state=TEST_SEED).reset_index(drop=True)
    return train, test, labels, desc


def data_hash(df: pd.DataFrame) -> str:
    return sha256(list(zip(df.text, df.label.astype(int))))


@contextmanager
def track_energy(name: str):
    """Measure wall time, energy (kWh) and emissions (kg CO2eq) of the enclosed block with CodeCarbon."""
    res = {"seconds": 0.0, "kwh": 0.0, "co2_kg": 0.0, "source": "codecarbon"}
    tracker = None
    try:
        from codecarbon import OfflineEmissionsTracker
        tracker = OfflineEmissionsTracker(
            project_name=name, country_iso_code=COUNTRY, measure_power_secs=1,
            save_to_file=False, log_level="error", allow_multiple_runs=True)
        tracker.start()
    except Exception as e:
        log.warning("CodeCarbon unavailable (%s)", e)
        tracker, res["source"] = None, "unavailable"
    t0 = time.perf_counter()
    try:
        yield res
    finally:
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        res["seconds"] = time.perf_counter() - t0
        if tracker is not None:
            try:
                tracker.stop()
                d = tracker.final_emissions_data
                res["kwh"], res["co2_kg"] = float(d.energy_consumed), float(d.emissions)
            except Exception as e:
                log.warning("CodeCarbon stop failed (%s)", e)
                res["source"] = "unavailable"


@dataclass
class TrainConfig:
    use_case: str
    base_model: str = DEFAULT_BASE
    n_per_class: int = 32
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 16
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    max_len: int = 128
    seed: int = 42


LORA_TARGETS = {"distilbert": ["q_lin", "v_lin"], "deberta-v2": ["query_proj", "value_proj"]}


def _encode(tok, texts, max_len):
    return tok(list(texts), truncation=True, max_length=max_len, padding=True, return_tensors="pt")


def create_model(cfg: TrainConfig, n_test: int = 300, evaluate: bool = True) -> dict:
    """Fine-tune `cfg.base_model` with LoRA on a few labelled examples, register it, then test it."""
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    set_seed(cfg.seed)
    train, _, labels, _ = load_task(cfg.use_case, cfg.n_per_class, 0, cfg.seed)
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model, num_labels=len(labels), id2label=dict(enumerate(labels)),
        label2id={l: i for i, l in enumerate(labels)}, ignore_mismatched_sizes=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        model.config.pad_token_id = tok.pad_token_id
    heads = [n for n, _ in model.named_children() if n in ("pre_classifier", "classifier", "score")]
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=LORA_TARGETS.get(model.config.model_type), modules_to_save=heads)).to(DEVICE)
    trainable, total = model.get_nb_trainable_parameters()

    enc = _encode(tok, train.text, cfg.max_len)
    loader = DataLoader(TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(train.label.values)),
                        batch_size=cfg.batch_size, shuffle=True, generator=torch.Generator().manual_seed(cfg.seed))
    steps = cfg.epochs * len(loader)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    with track_energy(f"train-{cfg.use_case}") as energy:
        for epoch in range(cfg.epochs):
            running = 0.0
            for ids, mask, y in loader:
                ids, mask, y = ids.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
                with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=use_amp):
                    loss = model(input_ids=ids, attention_mask=mask, labels=y).loss
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                running += loss.item()
            if epoch == 0 or (epoch + 1) % 5 == 0:
                log.info("epoch %d/%d loss %.4f", epoch + 1, cfg.epochs, running / len(loader))

    config = asdict(cfg)
    cfg_hash = sha256(config)
    model_id = f"slm-{cfg.use_case}-{cfg.base_model.split('/')[-1]}-{cfg.n_per_class}pc-{cfg_hash[:6]}"
    path = MODELS_DIR / model_id
    model.save_pretrained(path)
    tok.save_pretrained(path)
    _upsert("models", {
        "model_id": model_id, "use_case": cfg.use_case, "kind": "slm", "base_model": cfg.base_model,
        "n_per_class": cfg.n_per_class, "n_train": len(train), "trainable_params": trainable,
        "total_params": total, "path": str(path), "labels": labels,
        "config": {**config, "env": env_snapshot()}, "config_hash": cfg_hash, "data_hash": data_hash(train),
        "train_seconds": energy["seconds"], "train_kwh": energy["kwh"], "train_co2_kg": energy["co2_kg"],
        "created_at": now()})
    _CACHE[model_id] = (model.eval(), tok)
    log.info("registered %s (%d/%d trainable params, %.2fs, %.5f Wh)", model_id, trainable, total,
             energy["seconds"], energy["kwh"] * 1000)
    out = {"model_id": model_id, "n_train": len(train), "trainable_params": trainable, "total_params": total,
           "train_seconds": round(energy["seconds"], 2), "train_wh": round(energy["kwh"] * 1000, 5)}
    if evaluate:
        out["evaluation"] = test_model(model_id, n_test=n_test)
    return out


_CACHE: dict[str, tuple] = {}


def _load_slm(model_id: str):
    if model_id not in _CACHE:
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        m = get_model(model_id)
        labels = m["labels"]
        tok = AutoTokenizer.from_pretrained(m["path"])
        base = AutoModelForSequenceClassification.from_pretrained(
            m["base_model"], num_labels=len(labels), id2label=dict(enumerate(labels)),
            label2id={l: i for i, l in enumerate(labels)}, ignore_mismatched_sizes=True)
        if tok.pad_token_id is not None:
            base.config.pad_token_id = tok.pad_token_id
        _CACHE[model_id] = (PeftModel.from_pretrained(base, m["path"]).to(DEVICE).eval(), tok)
    return _CACHE[model_id]


@torch.no_grad()
def _slm_predict_ids(model_id: str, texts: list[str], batch_size: int = 64, max_len: int = 128) -> np.ndarray:
    model, tok = _load_slm(model_id)
    preds = []
    for i in range(0, len(texts), batch_size):
        enc = {k: v.to(DEVICE) for k, v in _encode(tok, texts[i:i + batch_size], max_len).items()}
        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
            preds.append(model(**enc).logits.argmax(-1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=int)


def predict(model_id: str, texts: list[str]) -> list[dict]:
    """Classify texts with a registered model (local SLM, local LLM or Gemini)."""
    m = get_model(model_id)
    if m["kind"] == "slm":
        ids = _slm_predict_ids(model_id, texts)
    else:
        ids, _ = _llm_predict_ids(m, texts)
    return [{"text": t, "label": m["labels"][i] if i >= 0 else None} for t, i in zip(texts, ids)]


def _score(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    y_pred = np.where(np.asarray(y_pred) < 0, -1, y_pred)
    labels = sorted(set(np.asarray(y_true).tolist()))
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))}


def _record_eval(model_id, use_case, test, y_pred, energy, left_premises=0, pii=0) -> dict:
    n = len(test)
    if n == 0:
        raise ValueError("empty test set")
    scores = _score(test.label.values, y_pred)
    row = {"model_id": model_id, "use_case": use_case, "n_test": n, "test_hash": data_hash(test), **scores,
           "seconds": energy["seconds"], "kwh": energy["kwh"], "co2_kg": energy["co2_kg"],
           "wh_per_1k": energy["kwh"] * 1000 / n * 1000, "latency_ms": energy["seconds"] / n * 1000,
           "energy_source": energy["source"], "data_left_premises": left_premises, "pii_redacted": pii,
           "created_at": now()}
    _upsert("evaluations", row)
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()
            if k not in ("test_hash", "created_at")}


def test_model(model_id: str, n_test: int = 300) -> dict:
    """Evaluate a registered model on the fixed test sample of its use case (energy-tracked)."""
    m = get_model(model_id)
    _, test, _, _ = load_task(m["use_case"], 0, n_test)
    if m["kind"] == "slm":
        _load_slm(model_id)
        with track_energy(f"eval-{model_id}") as energy:
            y_pred = _slm_predict_ids(model_id, test.text.tolist())
        return _record_eval(model_id, m["use_case"], test, y_pred, energy)
    y_pred, info = _llm_predict_ids(m, test.text.tolist())
    return _record_eval(model_id, m["use_case"], test, y_pred, info["energy"],
                        left_premises=int(info["remote"]), pii=info["pii"])


def _match_label(answer: str, labels: list[str]) -> int:
    a = answer.strip().strip("\"'.").lower()
    for i, l in enumerate(labels):
        if a == l.lower():
            return i
    for i, l in sorted(enumerate(labels), key=lambda x: -len(x[1])):
        if re.search(rf"\b{re.escape(l.lower())}\b", a):
            return i
    return -1


def _prompt(desc: str, labels: list[str], texts: list[str]) -> str:
    items = "\n".join(f"{i + 1}. {t[:600]}" for i, t in enumerate(texts))
    return (f"You are a text classifier. Task: {desc}.\nAllowed labels: {json.dumps(labels)}.\n"
            f"Classify each of the {len(texts)} numbered texts below. Reply ONLY with a JSON array of "
            f"{len(texts)} label strings, in the same order.\n\n{items}")


def _gemini_client():
    from google import genai
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY missing in .env")
    return genai.Client(api_key=key)


def _gemini_ids(model_name, desc, labels, texts, batch_size=25):
    from google.genai import types
    client = _gemini_client()
    ids, tokens, calls = [], 0, 0
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        for attempt in range(6):
            try:
                r = client.models.generate_content(
                    model=model_name, contents=_prompt(desc, labels, chunk),
                    config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"))
                break
            except Exception as e:
                if attempt == 5:
                    raise
                wait = 2 ** attempt * 5
                log.warning("Gemini error (%s), retrying in %ss", str(e)[:120], wait)
                time.sleep(wait)
        calls += 1
        tokens += getattr(r.usage_metadata, "total_token_count", 0) or 0
        try:
            answers = json.loads(r.text)
        except (json.JSONDecodeError, TypeError):
            answers = []
        if isinstance(answers, str):
            answers = [answers]
        elif isinstance(answers, dict):
            vals = list(answers.values())
            answers = next((v for v in vals if isinstance(v, list)), vals if all(isinstance(v, str) for v in vals) else [])
        if not isinstance(answers, list):
            answers = []
        answers = [str(a) for a in answers][:len(chunk)]
        answers += [""] * (len(chunk) - len(answers))
        ids += [_match_label(a, labels) for a in answers]
    return np.array(ids), calls, tokens


def _local_llm_ids(model_name, desc, labels, texts):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    key = f"llm::{model_name}"
    if key not in _CACHE:
        tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16 if DEVICE == "cuda" else torch.bfloat16).to(DEVICE).eval()
        _CACHE[key] = (model, tok)
    model, tok = _CACHE[key]
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": f"You classify texts. Task: {desc}. Answer with exactly one label from "
                                       f"{json.dumps(labels)} and nothing else."},
         {"role": "user", "content": t[:1500]}], tokenize=False, add_generation_prompt=True) for t in texts]
    ids = []
    with torch.no_grad():
        for i in range(0, len(prompts), 16):
            enc = tok(prompts[i:i + 16], return_tensors="pt", padding=True).to(DEVICE)
            out = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
            for seq in out[:, enc["input_ids"].shape[1]:]:
                ids.append(_match_label(tok.decode(seq, skip_special_tokens=True), labels))
    return np.array(ids)


def _llm_predict_ids(m: dict, texts: list[str]):
    """Run a registered LLM entry. Remote calls only ever see PII-redacted text."""
    desc = next(u["description"] for u in use_cases() if u["name"] == m["use_case"])
    labels, provider = m["labels"], m["config"]["provider"]
    if provider == "gemini":
        clean = [redact(t) for t in texts]
        pii = sum(n for _, n in clean)
        with track_energy(f"llm-{m['model_id']}") as energy:
            ids, calls, tokens = _gemini_ids(m["base_model"], desc, labels, [t for t, _ in clean])
        energy["kwh"] += calls * GEMINI_WH_PER_PROMPT / 1000
        energy["co2_kg"] += calls * GEMINI_G_CO2_PER_PROMPT / 1000
        energy["source"] = f"codecarbon(client)+estimate({GEMINI_WH_PER_PROMPT}Wh/prompt x {calls})"
        return ids, {"remote": True, "pii": pii, "energy": energy, "tokens": tokens}
    _local_llm_ids(m["base_model"], desc, labels, texts[:1])
    with track_energy(f"llm-{m['model_id']}") as energy:
        ids = _local_llm_ids(m["base_model"], desc, labels, texts)
    return ids, {"remote": False, "pii": 0, "energy": energy}


def run_llm_baseline(use_case: str, provider: str = "local", model_name: str | None = None, n_test: int = 300) -> dict:
    """Register a zero-shot LLM for a use case (no annotations needed) and evaluate it."""
    if provider not in ("local", "gemini"):
        raise ValueError("provider must be 'local' or 'gemini'")
    model_name = model_name or (os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite") if provider == "gemini"
                                else DEFAULT_LOCAL_LLM)
    labels = next((u["labels"] for u in use_cases() if u["name"] == use_case), None)
    if labels is None:
        raise KeyError(f"unknown use case {use_case!r}")
    model_id = f"llm-{provider}-{model_name.split('/')[-1]}-{use_case}"
    _upsert("models", {"model_id": model_id, "use_case": use_case, "kind": f"llm-{provider}",
                       "base_model": model_name, "n_per_class": 0, "n_train": 0, "trainable_params": 0,
                       "total_params": None, "path": None, "labels": labels,
                       "config": {"provider": provider, "zero_shot": True, "env": env_snapshot()},
                       "config_hash": sha256([provider, model_name]), "data_hash": None,
                       "train_seconds": 0, "train_kwh": 0, "train_co2_kg": 0, "created_at": now()})
    return {"model_id": model_id, "evaluation": test_model(model_id, n_test)}


def benchmark(use_case: str, n_per_class: int = 32, n_test: int = 300, providers=("local", "gemini"),
              base_model: str = DEFAULT_BASE, seed: int = 42) -> pd.DataFrame:
    """Full protocol for one use case: few-shot LoRA SLM vs zero-shot LLM(s) on the same test sample."""
    create_model(TrainConfig(use_case=use_case, base_model=base_model, n_per_class=n_per_class, seed=seed),
                 n_test=n_test)
    for p in providers:
        try:
            run_llm_baseline(use_case, p, n_test=n_test)
        except Exception as e:
            log.error("LLM baseline %s failed for %s: %s", p, use_case, e)
    lb = leaderboard(use_case)
    lb.to_csv(RESULTS_DIR / f"benchmark_{use_case}.csv", index=False)
    return lb


def breakeven(use_case: str) -> pd.DataFrame:
    """Predictions after which an SLM (training + inference) beats each LLM on total energy."""
    lb = leaderboard(use_case)
    rows = []
    for _, s in lb[lb.kind == "slm"].iterrows():
        for _, l in lb[lb.kind.str.startswith("llm")].iterrows():
            gain = l.wh_per_1k - s.wh_per_1k
            measured = s.energy_source != "unavailable"
            rows.append({"slm": s.model_id, "llm": l.model_id, "slm_train_wh": s.train_wh,
                         "breakeven_predictions": (int(np.ceil(s.train_wh / gain * 1000))
                                                   if gain > 0 and measured else None),
                         "energy_ratio_llm_over_slm": round(l.wh_per_1k / s.wh_per_1k, 1) if s.wh_per_1k else None})
    return pd.DataFrame(rows)


def write_report() -> Path:
    lb = leaderboard()
    path = RESULTS_DIR / "leaderboard.md"
    parts = ["# Frugal NLP leaderboard\n", f"_generated {now()} on {env_snapshot()['gpu'] or 'CPU'}_\n",
             lb.drop(columns=["energy_source"]).to_markdown(index=False)]
    for uc in lb.use_case.unique():
        be = breakeven(uc)
        if not be.empty:
            parts += [f"\n## Energy break-even: {uc}\n", be.to_markdown(index=False)]
    path.write_text("\n".join(parts), encoding="utf-8")
    lb.to_csv(RESULTS_DIR / "leaderboard.csv", index=False)
    return path


def _cli():
    p = argparse.ArgumentParser(description="Frugal NLP benchmark: few-shot LoRA SLMs vs LLMs")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="create a LoRA model for a use case")
    t.add_argument("--use-case", required=True)
    for f in TrainConfig.__dataclass_fields__.values():
        if f.name != "use_case":
            t.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    t.add_argument("--n-test", type=int, default=300)
    e = sub.add_parser("test", help="evaluate a registered model")
    e.add_argument("--model-id", required=True)
    e.add_argument("--n-test", type=int, default=300)
    l = sub.add_parser("llm", help="zero-shot LLM baseline")
    l.add_argument("--use-case", required=True)
    l.add_argument("--provider", choices=["local", "gemini"], default="local")
    l.add_argument("--model-name")
    l.add_argument("--n-test", type=int, default=300)
    b = sub.add_parser("benchmark", help="SLM vs LLM protocol (use-case or 'all')")
    b.add_argument("--use-case", default="all")
    b.add_argument("--n-per-class", type=int, default=32)
    b.add_argument("--n-test", type=int, default=300)
    b.add_argument("--providers", default="local,gemini")
    b.add_argument("--base-model", default=DEFAULT_BASE)
    pr = sub.add_parser("predict", help="classify texts")
    pr.add_argument("--model-id", required=True)
    pr.add_argument("text", nargs="+")
    r = sub.add_parser("register", help="add a private CSV use case (stored encrypted)")
    r.add_argument("--name", required=True)
    r.add_argument("--csv", required=True)
    r.add_argument("--text-col", required=True)
    r.add_argument("--label-col", required=True)
    r.add_argument("--description", default="")
    sub.add_parser("models", help="list registered models")
    sub.add_parser("use-cases", help="list use cases")
    lb = sub.add_parser("leaderboard", help="show the SQL leaderboard and write results/leaderboard.md")
    lb.add_argument("--use-case")
    rec = sub.add_parser("recommend", help="reference model for a use case under an energy budget")
    rec.add_argument("--use-case", required=True)
    rec.add_argument("--max-wh-per-1k", type=float)
    rec.add_argument("--min-f1", type=float, default=0.0)
    rec.add_argument("--f1-tolerance", type=float, default=0.02)
    rec.add_argument("--local-only", action="store_true")
    rd = sub.add_parser("redact", help="show PII redaction")
    rd.add_argument("text", nargs="+")
    a = p.parse_args()

    pd.set_option("display.width", 200, "display.max_columns", 30)
    show = lambda x: print(x.to_string(index=False) if isinstance(x, pd.DataFrame) else json.dumps(x, indent=2, default=str))
    if a.cmd == "train":
        kw = {k: getattr(a, k) for k in TrainConfig.__dataclass_fields__}
        show(create_model(TrainConfig(**kw), n_test=a.n_test))
    elif a.cmd == "test":
        show(test_model(a.model_id, a.n_test))
    elif a.cmd == "llm":
        show(run_llm_baseline(a.use_case, a.provider, a.model_name, a.n_test))
    elif a.cmd == "benchmark":
        for uc in (list(TASKS) if a.use_case == "all" else [a.use_case]):
            show(benchmark(uc, a.n_per_class, a.n_test, [x for x in a.providers.split(",") if x], a.base_model))
        print(f"\nreport: {write_report()}")
    elif a.cmd == "predict":
        show(predict(a.model_id, a.text))
    elif a.cmd == "register":
        show(register_use_case(a.name, a.csv, a.text_col, a.label_col, a.description))
    elif a.cmd == "models":
        show(list_models())
    elif a.cmd == "use-cases":
        show(use_cases())
    elif a.cmd == "leaderboard":
        show(leaderboard(a.use_case))
        print(f"\nreport: {write_report()}")
    elif a.cmd == "recommend":
        show(recommend(a.use_case, a.max_wh_per_1k, a.min_f1, a.f1_tolerance, a.local_only))
    elif a.cmd == "redact":
        show(dict(zip(("text", "n_redacted"), redact(" ".join(a.text)))))


if __name__ == "__main__":
    _cli()
