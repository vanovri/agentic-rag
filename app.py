"""Streamlit UI: ask a question, see the answer and the full agent trace."""
import streamlit as st

from agentic_rag.agent import Agent
from agentic_rag.config import Settings
from agentic_rag.llm import make_llm
from agentic_rag.tools import Retriever

st.set_page_config(page_title="Agentic RAG", layout="wide")
st.title("End-to-End Agentic RAG")
s = Settings()
s.provider = st.sidebar.selectbox("LLM provider", ["grok-cli", "openai"], index=0 if s.provider == "grok-cli" else 1)
st.sidebar.caption(f"Index: {s.index_dir}\nModel (openai): {s.model} @ {s.base_url}")


@st.cache_resource
def retriever():
    return Retriever(Settings())


q = st.text_input("Question", "How long can biochar carbon persist in soil?")
if st.button("Ask") and q:
    with st.spinner("Agent working (Grok CLI calls can take ~1 min each)..."):
        res = Agent(make_llm(s), retriever(), s).run(q)
    (st.success if res.found else st.warning)(res.answer)
    st.subheader("Agent trace")
    for t in res.trace:
        if t["step"] in ("tool", "re-retrieve", "plan"):
            with st.expander(f"{t['step']}: {t['tool']}({t['input']!r}) -> {t['summary']}"):
                if t.get("thought"): st.write("Thought:", t["thought"])
                for h in t.get("results", []):
                    st.markdown(f"**{h['id']}** (score {h['score']})\n\n> {h['text']}")
        elif t["step"] == "draft":
            with st.expander(f"draft round {t['round']}: {len(t['claims'])} claim(s)"):
                st.json(t["claims"])
        elif t["step"] == "verify":
            st.markdown(f"**Verification round {t['round']}**")
            for v in t["verdicts"]:
                st.write(("✅" if v["supported"] else "❌"), v["claim"], "—", v["reason"])
