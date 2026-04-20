import os

import requests
import streamlit as st
import time

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------ #
# Styles
# ------------------------------------------------------------------ #
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 4px;
}
.sub-title {
    text-align: center;
    color: #9ca3af;
    font-size: 16px;
    margin-bottom: 32px;
}
.preview-box {
    height: 560px;
    overflow-y: auto;
    padding: 24px 28px;
    border-radius: 12px;
    border: 1.5px solid #4CAF50;
    background-color: #0e1117;
    font-size: 15px;
    line-height: 1.75;
    color: #e5e7eb;
    white-space: pre-wrap;
    font-family: 'Georgia', serif;
}
.history-card {
    padding: 10px 14px;
    border-radius: 8px;
    border: 1px solid #374151;
    margin-bottom: 6px;
    background-color: #111827;
}
.stepper {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 20px 0;
    flex-wrap: wrap;
    gap: 4px;
}
.step { display: flex; align-items: center; font-weight: 600; font-size: 14px; }
.circle {
    width: 32px; height: 32px; border-radius: 50%;
    background: #374151; color: white;
    display: flex; justify-content: center; align-items: center;
    margin-right: 6px; font-size: 13px;
}
.active { background: #4CAF50 !important; }
.line { width: 60px; height: 3px; background: #374151; margin: 0 8px; }
.line.active { background: #4CAF50; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# Header
# ------------------------------------------------------------------ #
st.markdown("<div class='main-title'>📄 AI Research Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Multi-Agent IEEE Research Paper Generator powered by LangGraph</div>", unsafe_allow_html=True)
st.markdown("---")

# ------------------------------------------------------------------ #
# Session state
# ------------------------------------------------------------------ #
for key, default in [
    ("current_markdown", None),
    ("current_topic", None),
    ("history", []),
    ("progress_step", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ------------------------------------------------------------------ #
# Input form
# ------------------------------------------------------------------ #
with st.form("research_form"):
    topic = st.text_input(
        "🔬 Research Topic",
        placeholder="e.g., Federated Learning for Healthcare Data Privacy",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        max_papers = st.slider("Max papers to retrieve", 5, 25, 10)
        years_back = st.slider("Years back to search", 1, 8, 3)

    with col_b:
        st.markdown("**Author Information** *(used in LaTeX export)*")
        author_name = st.text_input("Full Name", value="Research Author", label_visibility="collapsed",
                                    placeholder="Full Name")
        author_institution = st.text_input("Institution", value="University", label_visibility="collapsed",
                                           placeholder="Institution / University")
        author_email = st.text_input("Email", value="author@university.edu", label_visibility="collapsed",
                                     placeholder="Email Address")

    run_btn = st.form_submit_button("🚀 Generate Research Paper", use_container_width=True)

# ------------------------------------------------------------------ #
# Stepper
# ------------------------------------------------------------------ #
STEP_LABELS = ["Retrieval", "RAG Ingestion", "Summarize", "Evaluate", "Design", "Report"]
stepper_placeholder = st.empty()


def render_stepper(step: int):
    html = "<div class='stepper'>"
    for i, label in enumerate(STEP_LABELS, start=1):
        active = "active" if step >= i else ""
        symbol = "✔" if step > i else str(i)
        html += f"<div class='step'><div class='circle {active}'>{symbol}</div>{label}</div>"
        if i < len(STEP_LABELS):
            html += f"<div class='line {active}'></div>"
    html += "</div>"
    stepper_placeholder.markdown(html, unsafe_allow_html=True)


render_stepper(st.session_state.progress_step)

# ------------------------------------------------------------------ #
# Run pipeline
# ------------------------------------------------------------------ #
if run_btn:
    if not topic.strip():
        st.warning("⚠️ Please enter a research topic.")
    else:
        st.session_state.progress_step = 1
        render_stepper(1)

        with st.spinner("Running full research pipeline — this takes 2-5 minutes..."):
            try:
                res = requests.post(
                    f"{API_URL}/run_pipeline",
                    json={
                        "topic": topic.strip(),
                        "max_papers": max_papers,
                        "years_back": years_back,
                        "author_name": author_name,
                        "author_institution": author_institution,
                        "author_email": author_email,
                    },
                    timeout=360,
                )
                data = res.json()
            except requests.exceptions.Timeout:
                st.error("⏱ Request timed out. The pipeline may still be running — try refreshing.")
                st.stop()
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Cannot connect to backend at `{API_URL}`. Make sure the backend is running (`uvicorn app:app --reload` inside the `backend/` folder).")
                st.stop()
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
                st.stop()

        if data.get("status") != "ok":
            st.error(f"Pipeline failed: {data.get('message', 'Unknown error')}")
            st.json(data)
            st.stop()

        session_id = data["session_id"]

        # Poll progress until complete
        for _ in range(120):
            time.sleep(1)
            try:
                p = requests.get(f"{API_URL}/progress/{session_id}", timeout=10).json()
                step = p.get("step", 1)
                st.session_state.progress_step = step
                render_stepper(step)
                if step >= 6:
                    break
            except Exception:
                pass

        render_stepper(6)
        st.success("✅ Research paper generated successfully!")

        md_content = data["report_markdown"]["content"]
        st.session_state.history.insert(0, {"topic": topic.strip(), "markdown": md_content})
        st.session_state.current_markdown = md_content
        st.session_state.current_topic = topic.strip()

# ------------------------------------------------------------------ #
# Load history from backend
# ------------------------------------------------------------------ #
st.markdown("---")
col_hist, col_refresh = st.columns([5, 1])
with col_refresh:
    if st.button("🔄 Refresh History"):
        try:
            res = requests.get(f"{API_URL}/history", timeout=20)
            st.session_state.history = res.json().get("history", [])
            st.success("History refreshed")
        except Exception as e:
            st.error(f"Could not load history: {e}")

with st.expander("📚 Previous Reports", expanded=False):
    if st.session_state.history:
        for idx, item in enumerate(st.session_state.history):
            c1, c2 = st.columns([6, 1])
            with c1:
                st.markdown(
                    f"<div class='history-card'>📄 <b>{item['topic']}</b></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("👁 View", key=f"view_{idx}"):
                    st.session_state.current_markdown = item["markdown"]
                    st.session_state.current_topic = item["topic"]
    else:
        st.info("No previous reports yet.")

# ------------------------------------------------------------------ #
# Paper preview
# ------------------------------------------------------------------ #
if st.session_state.current_markdown:
    st.markdown("---")
    st.markdown("<h2 style='text-align:center;'>📄 Research Paper Preview</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='preview-box'>{st.session_state.current_markdown}</div>",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------ #
# Download panel
# ------------------------------------------------------------------ #
if st.session_state.current_markdown and st.session_state.current_topic:
    st.markdown("---")
    st.subheader("⬇️ Download Report")
    col1, col2 = st.columns(2)

    with col1:
        try:
            md_res = requests.post(
                f"{API_URL}/download",
                json={"topic": st.session_state.current_topic},
                timeout=60,
            )
            st.download_button(
                label="📄 Download Markdown",
                data=md_res.content,
                file_name=f"{st.session_state.current_topic.replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Download failed: {e}")

    with col2:
        try:
            zip_res = requests.post(
                f"{API_URL}/download-zip",
                json={"topic": st.session_state.current_topic},
                timeout=60,
            )
            if zip_res.status_code != 200 or zip_res.headers.get("content-type", "").startswith("application/json"):
                err = zip_res.json().get("message", zip_res.text)
                st.error(f"ZIP generation failed: {err}")
            else:
                st.download_button(
                    label="📦 Download Overleaf ZIP",
                    data=zip_res.content,
                    file_name=f"{st.session_state.current_topic.replace(' ', '_')}_overleaf.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"ZIP download failed: {e}")
