"""Streamlit GUI for MTF — browser-based interface for My Theorist Friend."""

from __future__ import annotations

import asyncio
import queue
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from mtf.interface import HumanInterface

# ---------------------------------------------------------------------------
# Module-level session store — survives browser refreshes
# ---------------------------------------------------------------------------
# Streamlit session_state is reset on a full page refresh (F5 / reload).
# We keep a single-slot global mirror of the orchestrator's mutable state so
# that a refreshed tab can reconnect to a still-running analysis.
#
# Keys:  ui_queue, thread, messages, pending, running, finished,
#        final_report, error, tmp_dir
_live_session: dict[str, Any] = {}

# ---------------------------------------------------------------------------
# HumanInterface implementation
# ---------------------------------------------------------------------------

_REPLY_TIMEOUT = 3600.0  # 1 hour — abandon wait if browser tab was closed


def _blocking_get(rq: "queue.Queue[Any]") -> Any:
    """Wait on *rq* for up to *_REPLY_TIMEOUT* seconds.

    The orchestrator thread is a daemon thread, so it is automatically
    killed when the main process exits.  A finite timeout is still set so
    that a closed browser tab eventually releases the thread.
    """
    try:
        return rq.get(timeout=_REPLY_TIMEOUT)
    except queue.Empty:
        raise TimeoutError("No reply received from Streamlit within the timeout period.")


class StreamlitInterface(HumanInterface):
    """HumanInterface implementation that sends messages through a queue to Streamlit."""

    def __init__(self, ui_queue: "queue.Queue[tuple[str, Any, Any]]") -> None:
        self._q = ui_queue

    async def show(self, content: str, title: str = "") -> None:
        self._q.put(("show", content, title))

    async def ask(self, prompt: str) -> str:
        rq: queue.Queue[str] = queue.Queue()
        self._q.put(("ask", prompt, rq))
        return await asyncio.get_running_loop().run_in_executor(None, _blocking_get, rq)

    async def confirm(self, prompt: str) -> bool:
        rq: queue.Queue[bool] = queue.Queue()
        self._q.put(("confirm", prompt, rq))
        return await asyncio.get_running_loop().run_in_executor(None, _blocking_get, rq)

    async def ask_for_files(self) -> list[str]:
        # Files are supplied via st.file_uploader at startup and passed directly
        # to orchestrator.run(files=...), so this method is never called.
        return []


# ---------------------------------------------------------------------------
# Background orchestrator runner
# ---------------------------------------------------------------------------

def _run_orchestrator(
    phenomenon: str,
    files: list[str] | None,
    config: Any,
    ui_queue: "queue.Queue[tuple[str, Any, Any]]",
) -> None:
    from mtf.orchestrator import MTFOrchestrator

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    iface = StreamlitInterface(ui_queue)
    orch = MTFOrchestrator(config=config, interface=iface)
    try:
        report = loop.run_until_complete(orch.run(phenomenon, files=files or None))
        ui_queue.put(("done", report, None))
    except Exception:
        ui_queue.put(("error", traceback.format_exc(), None))
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

def _streamlit_app() -> None:
    import streamlit as st

    from mtf.config import MTFConfig

    st.set_page_config(page_title="My Theorist Friend", page_icon="🔬", layout="wide")
    st.title("🔬 My Theorist Friend (MTF)")
    st.caption("Multi-agent AI system for experimental physicists")

    ss = st.session_state

    # ------------------------------------------------------------------
    # Reconnect to a live session after a browser refresh
    # ------------------------------------------------------------------
    # On a full page reload st.session_state is blank. If the orchestrator
    # is still running (or finished) we restore its state so the user can
    # continue without losing progress.
    if not ss.get("_initialised") and _live_session:
        thread: threading.Thread | None = _live_session.get("thread")
        if thread is not None and (thread.is_alive() or _live_session.get("finished")):
            for k, v in _live_session.items():
                setattr(ss, k, v)
            ss._initialised = True
    elif not ss.get("_initialised"):
        ss._initialised = True

    # ------------------------------------------------------------------
    # Sidebar — configuration
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Configuration")
        n_literature = st.slider("Literature agents", 1, 6, 3)
        n_fitting = st.slider("Fitting agents", 1, 6, 3)
        n_reviewer = st.slider("Reviewer agents", 1, 6, 3)
        reviewer_verification_passes = st.slider(
            "Reviewer verification passes",
            min_value=1,
            max_value=3,
            value=1,
            help="Set >1 to run a structural 'check again' loop after the initial review (vibe-physics lesson: Claude stops at the first error found).",
        )
        reviewer_models_input = st.multiselect(
            "Reviewer models (optional diversity)",
            ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
            default=[],
            help="If set, reviewer agents cycle through these models. Leave empty to use the single Reviewer model below.",
        )
        n_proposal = st.slider("Proposal agents", 1, 4, 2)
        max_debate_rounds = st.slider("Max debate rounds", 1, 5, 3)
        skip_fitting = st.checkbox("Skip fitting phase (qualitative evaluation only)", value=False)
        n_qualitative = st.slider("Qualitative evaluation agents", 1, 6, 3)

        physics_domain_options = [
            "condensed_matter",
            "qft",
            "gr",
            "amo",
            "plasma",
            "nuclear",
            "particle_physics",
            "cosmology",
            "optics",
            "fluid_dynamics",
            "electromagnetism",
            "thermodynamics",
            "statistical_mechanics",
            "biophysics",
        ]
        physics_domains = st.multiselect(
            "Physics domains",
            physics_domain_options,
            default=["condensed_matter"],
        )
        reviewer_model = st.selectbox(
            "Reviewer model",
            ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
            index=1,
        )
        proposal_model = st.selectbox(
            "Proposal model",
            ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
            index=0,
        )
        enable_gpd = st.checkbox("Enable GPD physics verification", value=True)
        auto_detect_domains = st.checkbox(
            "Auto-detect physics domains",
            value=False,
            help="Use GPD route_protocol/route_skill to infer physics domains from the phenomenon text. Off by default — set domains manually above.",
        )
        enhanced_pdf = st.checkbox("Enhanced PDF extraction (2-pass figure analysis)", value=True)

        st.divider()
        st.subheader("Integrity & verification")
        citation_verification = st.checkbox(
            "Citation verification (re-check author/year/venue)",
            value=True,
            help="LiteratureAgent re-verifies up to 10 citations via search tools. Unconfirmed ones are flagged [UNVERIFIED].",
        )
        fitting_integrity_check = st.checkbox(
            "Fitting integrity check (detect fabricated results)",
            value=True,
            help="Wraps lmfit optimizer in a sentinel to detect whether a real fit was run. Warns if result appears hardcoded.",
        )

    # ------------------------------------------------------------------
    # Pre-run view
    # ------------------------------------------------------------------
    if not ss.get("running") and not ss.get("finished"):
        st.subheader("Describe your experiment")
        phenomenon = st.text_area(
            "Experimental phenomenon",
            height=150,
            placeholder="E.g. We observe a sharp resistance drop to zero at 92 K in a YBCO sample...",
        )
        uploaded_files = st.file_uploader(
            "Optional: upload images or PDFs (PNG, JPG, GIF, WebP, PDF)",
            accept_multiple_files=True,
            type=["png", "jpg", "jpeg", "gif", "webp", "pdf"],
        )

        if st.button("Run Analysis", type="primary", disabled=not phenomenon.strip()):
            # Save uploaded files to a temp directory
            tmp_dir = tempfile.TemporaryDirectory()
            uploaded_paths: list[str] = []
            for uf in uploaded_files or []:
                dest = Path(tmp_dir.name) / uf.name
                dest.write_bytes(uf.read())
                uploaded_paths.append(str(dest))

            config = MTFConfig(
                n_literature=n_literature,
                n_fitting=n_fitting,
                n_qualitative=n_qualitative,
                n_reviewer=n_reviewer,
                n_proposal=n_proposal,
                max_debate_rounds=max_debate_rounds,
                physics_domains=physics_domains or ["condensed_matter"],
                enable_gpd_mcp=enable_gpd,
                auto_detect_domains=auto_detect_domains,
                fitting_enabled=not skip_fitting,
                reviewer_model=reviewer_model,
                proposal_model=proposal_model,
                pdf_enhanced_extraction=enhanced_pdf,
                citation_verification=citation_verification,
                fitting_result_integrity_check=fitting_integrity_check,
                reviewer_verification_passes=reviewer_verification_passes,
                reviewer_models=reviewer_models_input,
            )

            ui_queue: "queue.Queue[tuple[str, Any, Any]]" = queue.Queue()
            thread = threading.Thread(
                target=_run_orchestrator,
                args=(phenomenon.strip(), uploaded_paths or None, config, ui_queue),
                daemon=True,
            )
            thread.start()

            ss.running = True
            ss.finished = False
            ss.ui_queue = ui_queue
            ss.thread = thread
            ss.messages = []
            ss.pending = None
            ss.final_report = None
            ss.error = None
            ss.tmp_dir = tmp_dir
            ss._initialised = True
            # Persist in the module-level store so a browser refresh can
            # reconnect to this run without losing progress.
            _live_session.update(
                running=True,
                finished=False,
                ui_queue=ui_queue,
                thread=thread,
                messages=ss.messages,  # shared list reference
                pending=None,
                final_report=None,
                error=None,
                tmp_dir=tmp_dir,
            )
            st.rerun()
        return

    # ------------------------------------------------------------------
    # Running / finished view
    # ------------------------------------------------------------------
    ui_queue = ss.get("ui_queue")

    # Drain the queue
    if ui_queue is not None and ss.get("running") and not ss.get("finished"):
        while True:
            try:
                msg = ui_queue.get_nowait()
            except queue.Empty:
                break
            kind = msg[0]
            if kind == "show":
                _, content, title = msg
                new_msg = {"type": "show", "content": content, "title": title or "MTF"}
                ss.messages = ss.get("messages", []) + [new_msg]
                _live_session["messages"] = ss.messages
            elif kind in ("ask", "confirm"):
                _, prompt, reply_q = msg
                ss.pending = {"type": kind, "prompt": prompt, "reply_q": reply_q}
                _live_session["pending"] = ss.pending
                break  # wait for user response before draining further
            elif kind == "done":
                _, report, _ = msg
                ss.final_report = report
                ss.running = False
                ss.finished = True
                _live_session.update(final_report=report, running=False, finished=True)
                if tmp_dir := ss.get("tmp_dir"):
                    tmp_dir.cleanup()
                    ss.tmp_dir = None
                    _live_session["tmp_dir"] = None
                break
            elif kind == "error":
                _, tb, _ = msg
                ss.error = tb
                ss.running = False
                ss.finished = True
                _live_session.update(error=tb, running=False, finished=True)
                if tmp_dir := ss.get("tmp_dir"):
                    tmp_dir.cleanup()
                    ss.tmp_dir = None
                    _live_session["tmp_dir"] = None
                break

    # Render accumulated show() messages
    messages = ss.get("messages", [])
    if messages:
        st.subheader("Analysis progress")
        for i, m in enumerate(messages):
            with st.expander(m["title"], expanded=(i == len(messages) - 1)):
                st.markdown(m["content"])

    # Render pending interaction widget
    pending = ss.get("pending")
    if pending is not None:
        st.divider()
        if pending["type"] == "ask":
            st.info(f"**MTF asks:** {pending['prompt']}")
            with st.form("ask_form", clear_on_submit=True):
                user_text = st.text_area("Your response", height=100)
                if st.form_submit_button("Submit"):
                    pending["reply_q"].put(user_text)
                    ss.pending = None
                    _live_session["pending"] = None
                    st.rerun()

        elif pending["type"] == "confirm":
            st.info(f"**MTF asks for approval:** {pending['prompt']}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Approve", type="primary"):
                    pending["reply_q"].put(True)
                    ss.pending = None
                    _live_session["pending"] = None
                    st.rerun()
            with col2:
                if st.button("✏️ Request Changes"):
                    pending["reply_q"].put(False)
                    ss.pending = None
                    _live_session["pending"] = None
                    st.rerun()

    # Poll while running with no pending interaction
    if ss.get("running") and not ss.get("finished") and ss.get("pending") is None:
        with st.spinner("Analysis in progress…"):
            time.sleep(0.3)
        st.rerun()

    # Final state
    if ss.get("finished"):
        if ss.get("final_report"):
            st.divider()
            st.success("Analysis complete!")
            st.subheader("Final Report")
            st.markdown(ss.final_report)
            st.download_button(
                label="Download report as Markdown",
                data=ss.final_report,
                file_name="mtf_report.md",
                mime="text/markdown",
            )
        elif ss.get("error"):
            st.divider()
            st.error("MTF encountered an error:")
            st.code(ss.error, language="text")

        if st.button("Start new analysis"):
            if tmp_dir := ss.get("tmp_dir"):
                tmp_dir.cleanup()
            for key in ["running", "finished", "ui_queue", "thread", "messages",
                        "pending", "final_report", "error", "tmp_dir", "_initialised"]:
                ss.pop(key, None)
            _live_session.clear()
            st.rerun()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the Streamlit app via `mtf-gui`."""
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())],
        check=True,
    )


# When Streamlit executes this file directly (`streamlit run gui.py`), run the app.
# Streamlit sets __name__ == "__main__" when running a script.
if __name__ == "__main__":
    _streamlit_app()
