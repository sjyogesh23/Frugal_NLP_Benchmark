"""Streamlit dashboard over the model registry: `streamlit run app.py`."""
import os
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import frugal_nlp as fn

st.set_page_config(page_title="Frugal NLP Benchmark", layout="wide")
st.title("Frugal NLP Benchmark")
st.caption(f"Small LoRA models vs LLMs · device: {fn.DEVICE} · registry: {fn.DB_PATH.name}")

tab_lb, tab_train, tab_llm, tab_pred, tab_priv = st.tabs(
    ["Leaderboard", "Create model", "LLM baseline", "Predict", "Privacy"])
cases = [u["name"] for u in fn.use_cases()]

with tab_lb:
    uc = st.selectbox("Use case", ["(all)"] + cases)
    lb = fn.leaderboard(None if uc == "(all)" else uc)
    if lb.empty:
        st.info("No evaluations yet: create a model or run `python frugal_nlp.py benchmark`.")
    else:
        st.dataframe(lb, width="stretch", hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 4))
            pts = lb[lb.wh_per_1k > 0]
            for kind, g in pts.groupby("kind"):
                ax.scatter(g.wh_per_1k, g.macro_f1, label=kind, s=70)
                for _, r in g.iterrows():
                    ax.annotate(r.use_case, (r.wh_per_1k, r.macro_f1), fontsize=8, xytext=(4, 4),
                                textcoords="offset points")
            ax.set_xscale("log")
            ax.set_xlabel("Wh per 1k predictions (log)")
            ax.set_ylabel("macro-F1")
            ax.set_title("Accuracy vs energy")
            ax.legend()
            st.pyplot(fig)
        with c2:
            if uc != "(all)":
                st.subheader("Energy break-even")
                st.dataframe(fn.breakeven(uc), hide_index=True)
                st.subheader("Recommended model")
                budget = st.number_input("Max Wh per 1k predictions (0 = no limit)", 0.0, value=0.0)
                st.json(fn.recommend(uc, budget or None))

with tab_train:
    with st.form("train"):
        c1, c2, c3 = st.columns(3)
        uc_t = c1.selectbox("Use case", cases, key="uc_t")
        n_pc = c2.number_input("Labels per class", 4, 1000, 32)
        base = c3.text_input("Base model", fn.DEFAULT_BASE)
        epochs = c1.number_input("Epochs", 1, 100, 20)
        r = c2.number_input("LoRA rank", 1, 64, 8)
        seed = c3.number_input("Seed", 0, 10_000, 42)
        if st.form_submit_button("Train + evaluate"):
            with st.spinner("Training with LoRA (energy tracked by CodeCarbon)…"):
                st.json(fn.create_model(fn.TrainConfig(use_case=uc_t, n_per_class=int(n_pc), base_model=base,
                                                       epochs=int(epochs), lora_r=int(r), seed=int(seed))))
    st.subheader("Registered models")
    st.dataframe(fn.list_models(), width="stretch", hide_index=True)
    with st.expander("Register a private CSV use case"):
        up = st.file_uploader("CSV", type="csv")
        if up is not None:
            df = pd.read_csv(up, dtype=str)
            name = st.text_input("Use-case name")
            tc = st.selectbox("Text column", df.columns)
            lc = st.selectbox("Label column", df.columns)
            if st.button("Register (stored encrypted)") and name:
                fd, tmp_path = tempfile.mkstemp(suffix=".csv")
                os.close(fd)
                try:
                    df.to_csv(tmp_path, index=False)
                    st.json(fn.register_use_case(name, tmp_path, tc, lc))
                except Exception as e:
                    st.error(str(e))
                finally:
                    os.unlink(tmp_path)

with tab_llm:
    c1, c2, c3 = st.columns(3)
    uc_l = c1.selectbox("Use case", cases, key="uc_l")
    prov = c2.selectbox("Provider", ["local", "gemini"])
    n_test = c3.number_input("Test size", 20, 2000, 300)
    if st.button("Run zero-shot baseline"):
        with st.spinner("Running LLM…"):
            try:
                st.json(fn.run_llm_baseline(uc_l, prov, n_test=int(n_test)))
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")

with tab_pred:
    models = fn.list_models()
    if models.empty:
        st.info("No models registered yet.")
    else:
        mid = st.selectbox("Model", models.model_id)
        text = st.text_area("One text per line", "I loved this movie!\nWorst purchase ever.")
        if st.button("Predict"):
            st.dataframe(pd.DataFrame(fn.predict(mid, [t for t in text.splitlines() if t.strip()])),
                         hide_index=True)

with tab_priv:
    st.write("Texts are anonymised with these rules before any call to a remote LLM. Small models run locally, "
             "so private data never leaves the machine; custom datasets are Fernet-encrypted at rest.")
    sample = st.text_area("Text", "Hi, I'm John (john.smith@acme.com, +91 98765 43210). Card 4111 1111 1111 1111.")
    clean, n = fn.redact(sample)
    st.code(clean)
    st.metric("PII items redacted", n)
