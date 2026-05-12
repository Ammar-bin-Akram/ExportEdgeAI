"""
ExportEdge AI — Mango Quality & Export Advisory Dashboard
=========================================================
Streamlit interface with a polished, professional UI.
"""

import streamlit as st

# ── Page config (MUST be first Streamlit call) ────────────────────────
st.set_page_config(
    page_title="ExportEdge AI — Mango Inspector",
    page_icon="🥭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS for professional styling ───────────────────────────────
st.markdown("""
<style>
/* ── Global overrides ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* ── Dark background ──────────────────────────────────────────── */
html, body {
    background-color: #ffffff !important;
}
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main, .block-container {
    background-color: #ffffff !important;
}
[data-testid="stHeader"] {
    background-color: #ffffff !important;
}

/* Hide default Streamlit header & footer */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* ── Sidebar styling ──────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #064e3b 0%, #022c22 100%);
    color: #d1fae5;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #6ee7b7;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li {
    color: #a7f3d0;
}

/* ── Metric card styling ──────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #f0fdf4 0%, #e6ffea 100%);
    border: 1px solid #10b981;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 12px rgba(16,185,129,0.1);
}
div[data-testid="stMetric"] label {
    color: #047857 !important;
    font-weight: 500;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #065f46 !important;
    font-weight: 700;
}

/* ── Button styling ───────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.6rem 1.5rem;
    transition: all 0.2s ease;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #047857 0%, #059669 100%);
    box-shadow: 0 4px 16px rgba(16,185,129,0.3);
}
.stButton > button {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #047857 0%, #059669 100%) !important;
    box-shadow: 0 4px 16px rgba(16,185,129,0.3) !important;
}

/* ── Expander styling ─────────────────────────────────────────── */
div[data-testid="stExpander"] {
    border: 1px solid #d1d5db;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 8px;
}
div[data-testid="stExpander"] summary {
    font-weight: 600;
    display: block;
    padding: 12px 16px;
    background-color: #f3f4f6;
    margin-bottom: 8px;
    border-bottom: 1px solid #e5e7eb;
    border-radius: 8px 8px 0 0;
    color: #1f2937;
}

/* ── Dataframe styling ────────────────────────────────────────── */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ── Custom hero section ──────────────────────────────────────── */
.hero-container {
    text-align: center;
    padding: 2rem 1rem;
}
.hero-container h1 {
    font-size: 4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #10b981, #6ee7b7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.hero-container p.subtitle {
    font-size: 1.2rem;
    color: #475569;
    margin-bottom: 2rem;
}

/* ── Feature cards ────────────────────────────────────────────── */
.feature-card {
    background: linear-gradient(135deg, #f0fdf4 0%, #e6ffea 100%);
    border: 1.5px solid #10b981;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    min-height: 160px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.feature-card:hover {
    transform: translateY(-2px);
    border-color: #059669;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
}
.feature-card .icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: center;
    align-items: center;
}
.feature-card h3 {
    color: #065f46;
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 0.3rem;
}
.feature-card p {
    color: #047857;
    font-size: 0.85rem;
    line-height: 1.4;
}

/* ── Grade badge ──────────────────────────────────────────────── */
.grade-badge {
    display: inline-block;
    padding: 4px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.9rem;
}
.grade-a { background: #065f46; color: #6ee7b7; }
.grade-b { background: #78350f; color: #fde68a; }
.grade-c { background: #7f1d1d; color: #fca5a5; }

/* ── Section divider ──────────────────────────────────────────── */
.section-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #d1d5db, transparent);
    margin: 1.5rem 0;
}

/* ── Status pills ─────────────────────────────────────────────── */
.status-pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
}
.status-healthy { background: #065f46; color: #6ee7b7; }
.status-diseased { background: #7f1d1d; color: #fca5a5; }
.status-minimal { background: #065f46; color: #6ee7b7; }
.status-moderate { background: #78350f; color: #fde68a; }
.status-significant { background: #7f1d1d; color: #fca5a5; }

/* ── Chat page: message bubbles ───────────────────────────────── */
.chat-wrapper {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 16px 16px 0 0;
    padding: 24px 16px 16px 16px;
    min-height: 50vh;
    max-height: 62vh;
    overflow-y: auto;
    margin-bottom: 0;
}
.chat-input-bar {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-top: none;
    border-radius: 0 0 16px 16px;
    padding: 12px 16px;
    margin-bottom: 1rem;
}
.chat-bubble {
    padding: 12px 18px;
    border-radius: 16px;
    margin-bottom: 14px;
    max-width: 75%;
    line-height: 1.6;
    font-size: 0.95rem;
    word-wrap: break-word;
    clear: both;
}
.chat-bubble p { margin: 0 0 6px 0; }
.chat-bubble p:last-child { margin-bottom: 0; }
.chat-user {
    background: linear-gradient(135deg, #065f46, #064e3b);
    color: #d1fae5;
    border: 1px solid #10b981;
    border-bottom-right-radius: 4px;
    float: right;
    text-align: right;
}
.chat-user .chat-label {
    font-size: 0.7rem;
    color: #6ee7b7;
    margin-bottom: 4px;
    font-weight: 600;
}
.chat-assistant {
    background: linear-gradient(135deg, #f0fdf4 0%, #e6ffea 100%);
    color: #1f2937;
    border: 1px solid #10b981;
    border-bottom-left-radius: 4px;
    float: left;
}
.chat-assistant .chat-label {
    font-size: 0.7rem;
    color: #047857;
    margin-bottom: 4px;
    font-weight: 600;
}
.chat-clearfix { clear: both; }
.chat-empty {
    text-align: center;
    color: #047857;
    padding: 3rem 1rem;
    font-size: 1rem;
}
.chat-empty .icon { font-size: 2.5rem; margin-bottom: 0.5rem; }

/* ── Global text color for white background ──────────────────── */
[class*="st-"] {
    color: #1f2937 !important;
}
.stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
    color: #1f2937 !important;
}

/* ── Form submit button styling ───────────────────────────────── */
.stFormSubmitButton > button {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
.stFormSubmitButton > button:hover {
    background: linear-gradient(135deg, #047857 0%, #059669 100%) !important;
    box-shadow: 0 4px 16px rgba(16,185,129,0.3) !important;
}

/* ── File uploader styling - more aggressive ──────────────────── */
[data-testid="stFileUploadDropzone"],
[data-testid="stFileUploadDropzone"] * {
    background-color: transparent !important;
}

div[data-testid="stFileUploadDropzone"] {
    background: linear-gradient(135deg, #f0fdf4 0%, #e6ffea 100%) !important;
    border: 2px dashed #10b981 !important;
    border-radius: 12px !important;
    padding: 24px !important;
}

/* ── File uploader text styling ──────────────────────────────── */
div[data-testid="stFileUploadDropzone"] p,
div[data-testid="stFileUploadDropzone"] div,
div[data-testid="stFileUploadDropzone"] span {
    color: #047857 !important;
}

/* ── File uploader button - all possible selectors ─────────────── */
[data-testid="stFileUploadDropzone"] button,
[data-testid="stFileUploadDropzone"] input[type="button"],
.stFileUploadWidget button {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
}

[data-testid="stFileUploadDropzone"] button:hover,
[data-testid="stFileUploadDropzone"] input[type="button"]:hover,
.stFileUploadWidget button:hover {
    background: linear-gradient(135deg, #047857 0%, #059669 100%) !important;
    box-shadow: 0 4px 16px rgba(16,185,129,0.3) !important;
}

/* ── Input fields styling ─────────────────────────────────────── */
input[type="text"], input[type="number"], input[type="email"], input[type="password"], textarea, select {
    background-color: #f9fafb !important;
    border: 1px solid #d1d5db !important;
    color: #1f2937 !important;
    border-radius: 8px !important;
}
input[type="text"]:focus, input[type="number"]:focus, input[type="email"]:focus, input[type="password"]:focus, textarea:focus, select:focus {
    border-color: #10b981 !important;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1) !important;
}

/* ── Label styling ────────────────────────────────────────────── */
label {
    color: #1f2937 !important;
    font-weight: 500 !important;
}

/* ── Logo styling ────────────────────────────────────────────── */
[data-testid="stImage"] img {
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
    transition: all 0.3s ease;
}
[data-testid="stImage"] img:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(16, 185, 129, 0.25);
}
</style>
""", unsafe_allow_html=True)

# ── Heavy imports (show spinner) ──────────────────────────────────────
with st.spinner("Loading dependencies..."):
    import cv2
    import numpy as np
    import os
    import time
    import tempfile
    from datetime import datetime

    from config.settings import Settings
    from vision import ROIExtractor, MotionDetector
    from vision.defect_detector import MangoDefectDetector
    from vision.mask_utils import create_mango_hsv_mask
    from vision.export_advisor import ExportAdvisor, parse_structured_recommendation
    from vision.report_generator import ReportGenerator
    from vision.export_report_repository import ExportReportRepository
    from vision.inspection_logger import log_inspection
    from language.comtrade_price_service import ComtradePriceService
    from language.report_embeddings import ReportEmbeddingService

    import config
    from video_utils import (
        create_background_subtractor,
        extract_roi,
        process_frame_for_motion,
    )
    from preprocessing import preprocess_image
    from model_utils import load_model_weights, predict_disease
    from segmentation_utils import (
        load_segmentation_model,
        segment_disease,
        get_disease_statistics,
    )

# ── Session state initialisation ──────────────────────────────────────
_defaults = {
    "started": False,
    "model_loaded": False,
    "model": None,
    "segmentation_model_loaded": False,
    "segmentation_model": None,
    "defect_detector": MangoDefectDetector(),
    "export_advisor": ExportAdvisor(),
    "price_service": ComtradePriceService(),
    "report_generator": ReportGenerator(),
    "report_repository": ExportReportRepository(),
    "report_embedding_service": ReportEmbeddingService(),
    "active_report_json": None,
    "source_type": "Video File",
    "live_max_seconds": 150,
    "live_max_events": 6,
    "live_cap": None,
    "live_fgbg": None,
    "live_frame_idx": 0,
    "live_in_motion": False,
    "live_motion_buffer": [],
    "live_low_motion_counter": 0,
    "live_detected_frames": [],
    "live_start_time": 0.0,
    "live_stop_requested": False,
    "processing_complete": False,
    "last_processing_error": None,
    "results": [],
    "chat_messages": [],
    "chat_rag_ready": False,
    "chat_retriever": None,
    "chat_llm": None,
    "current_page": "home",
}
for key, default in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Cached model loaders ──────────────────────────────────────────────
@st.cache_resource
def initialize_model():
    return load_model_weights()


@st.cache_resource
def initialize_segmentation_model():
    try:
        return load_segmentation_model()
    except Exception:
        return None


if not st.session_state.model_loaded:
    try:
        st.session_state.model = initialize_model()
        st.session_state.model_loaded = True
    except Exception:
        pass

if not st.session_state.segmentation_model_loaded:
    try:
        st.session_state.segmentation_model = initialize_segmentation_model()
        st.session_state.segmentation_model_loaded = True
    except Exception:
        pass


def load_model():
    if not st.session_state.model_loaded:
        if st.session_state.model is None:
            with st.spinner("Loading model..."):
                st.session_state.model = initialize_model()
                st.session_state.model_loaded = True


def _reset_live_runtime_state():
    """Release and clear live-capture runtime state."""
    cap = st.session_state.get("live_cap")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass

    st.session_state.live_cap = None
    st.session_state.live_fgbg = None
    st.session_state.live_frame_idx = 0
    st.session_state.live_in_motion = False
    st.session_state.live_motion_buffer = []
    st.session_state.live_low_motion_counter = 0
    st.session_state.live_detected_frames = []
    st.session_state.live_start_time = 0.0
    st.session_state.live_stop_requested = False


def _is_camera_source_reachable(camera_source, timeout_sec=5) -> bool:
    """Quick validation that a camera source can be opened and read with timeout."""
    import threading
    
    result = {"ok": False}
    
    def try_open_and_read():
        try:
            cap = cv2.VideoCapture(camera_source)
            # Set buffer size to minimum to reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                return
            ok, _ = cap.read()
            cap.release()
            result["ok"] = bool(ok)
        except Exception:
            pass
    
    thread = threading.Thread(target=try_open_and_read, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)
    
    return result["ok"]


def _sources_from_config_py() -> tuple[str, str]:
    """Load VIDEO_SOURCE and CAMERA_SOURCE explicitly from Code/config.py."""
    try:
        import importlib.util

        config_py_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
        if not os.path.exists(config_py_path):
            return ""

        spec = importlib.util.spec_from_file_location("project_config_py", config_py_path)
        if spec is None or spec.loader is None:
            return "", ""

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        video_source = str(getattr(module, "VIDEO_SOURCE", "")).strip()
        camera_source = str(getattr(module, "CAMERA_SOURCE", "")).strip()
        return video_source, camera_source
    except Exception:
        return "", ""


def _process_live_video_cycle(
    video_source,
    video_col,
    frame_col,
    status_placeholder,
    source_label,
    max_live_seconds,
    max_live_events,
    frames_per_cycle=50,
):
    """Process a short chunk of live frames so UI controls (Stop button) stay responsive."""
    if st.session_state.live_cap is None:
        cap = cv2.VideoCapture(video_source)
        # Reduce buffer to minimize latency and timeout
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            st.session_state.last_processing_error = (
                f"Could not open camera source: {source_label}"
            )
            status_placeholder.error(st.session_state.last_processing_error)
            return True, []
        st.session_state.live_cap = cap
        st.session_state.live_fgbg = create_background_subtractor()
        st.session_state.live_start_time = time.monotonic()

    cap = st.session_state.live_cap
    fgbg = st.session_state.live_fgbg

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(raw_fps) if raw_fps and raw_fps > 1 else 30

    video_placeholder = video_col.empty()
    frame_placeholder = frame_col.empty()
    info_placeholder = frame_col.empty()
    prediction_placeholder = frame_col.empty()

    done_reason = None
    frames_processed = 0
    consecutive_read_failures = 0

    while frames_processed < frames_per_cycle:
        elapsed = time.monotonic() - st.session_state.live_start_time
        if st.session_state.live_stop_requested:
            done_reason = "stopped by user"
            break
        if elapsed >= max_live_seconds:
            done_reason = "time window reached"
            break
        if len(st.session_state.live_detected_frames) >= max_live_events:
            done_reason = "max detections reached"
            break

        ret, frame = cap.read()
        if not ret:
            consecutive_read_failures += 1
            # If 3 consecutive reads fail, assume stream is dead
            if consecutive_read_failures >= 3:
                done_reason = "camera stream timeout or disconnected"
                break
            continue
        
        consecutive_read_failures = 0

        st.session_state.live_frame_idx += 1
        frame_idx = st.session_state.live_frame_idx
        frames_processed += 1

        roi = extract_roi(frame)
        if roi.size == 0:
            continue

        _, motion_area = process_frame_for_motion(frame, roi, fgbg)
        timestamp_ms = int((time.monotonic() - st.session_state.live_start_time) * 1000)

        if frame_idx % 5 == 0:
            display_frame = frame.copy()
            x1, y1, x2, y2 = config.ROI_COORDS
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"Live Frame {frame_idx}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            video_placeholder.image(
                cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                channels="RGB", width="stretch",
            )

        if (not st.session_state.live_in_motion) and motion_area > config.MOTION_AREA_THRESHOLD:
            st.session_state.live_in_motion = True
            st.session_state.live_motion_buffer = []
            st.session_state.live_low_motion_counter = 0

        if st.session_state.live_in_motion:
            st.session_state.live_motion_buffer.append((frame.copy(), motion_area, frame_idx, timestamp_ms))
            if motion_area < config.MOTION_AREA_THRESHOLD:
                st.session_state.live_low_motion_counter += 1
            else:
                st.session_state.live_low_motion_counter = 0

            if st.session_state.live_low_motion_counter >= config.MOTION_END_FRAMES:
                st.session_state.live_in_motion = False
                if st.session_state.live_motion_buffer:
                    result_data = _process_peak_frame(
                        st.session_state.live_motion_buffer,
                        frame_col,
                        frame_placeholder,
                        info_placeholder,
                        prediction_placeholder,
                    )
                    st.session_state.live_detected_frames.append(result_data)
                st.session_state.live_motion_buffer = []
                st.session_state.live_low_motion_counter = 0

    elapsed = int(time.monotonic() - st.session_state.live_start_time)

    if done_reason:
        # Finalize any active motion event on stop/timeout.
        if (
            st.session_state.live_in_motion
            and st.session_state.live_motion_buffer
            and len(st.session_state.live_detected_frames) < max_live_events
        ):
            result_data = _process_peak_frame(
                st.session_state.live_motion_buffer,
                frame_col,
                frame_placeholder,
                info_placeholder,
                prediction_placeholder,
            )
            st.session_state.live_detected_frames.append(result_data)

        results = list(st.session_state.live_detected_frames)
        _reset_live_runtime_state()
        if done_reason in ("camera stream ended", "camera stream timeout or disconnected") and not results:
            st.session_state.last_processing_error = (
                "Camera feed is not available or got disconnected. "
                "Please check CAMERA_SOURCE/camera connection and network status."
            )
            status_placeholder.error(st.session_state.last_processing_error)
        else:
            status_placeholder.success(
                f"Live processing complete — {len(results)} mango(es) detected in {elapsed}s ({done_reason})"
            )
        return True, results

    status_placeholder.info(
        f"Live processing from {source_label} @ ~{fps} FPS · "
        f"elapsed {elapsed}s · detections {len(st.session_state.live_detected_frames)}/{max_live_events}"
    )
    return False, list(st.session_state.live_detected_frames)


# ======================================================================
#   HELPER: grade badge HTML
# ======================================================================
def _grade_badge(impact: str) -> str:
    css = {"minimal": "grade-a", "moderate": "grade-b", "significant": "grade-c"}.get(
        impact, "grade-b"
    )
    label = {"minimal": "A", "moderate": "B", "significant": "C"}.get(impact, "?")
    return f'<span class="grade-badge {css}">Grade {label}</span>'


def _impact_pill(impact: str) -> str:
    css = {"minimal": "status-minimal", "moderate": "status-moderate",
           "significant": "status-significant"}.get(impact, "status-moderate")
    return f'<span class="status-pill {css}">{impact.title()}</span>'


# ======================================================================
#   VIDEO PROCESSING (unchanged logic, cleaner status messages)
# ======================================================================
def process_video_with_ui(
    video_source,
    video_col,
    frame_col,
    status_placeholder,
    live_mode=False,
    source_label="Video",
    max_live_seconds=45,
    max_live_events=6,
):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        status_placeholder.error(f"Could not open source: {source_label}")
        return []

    fgbg = create_background_subtractor()

    frame_idx = 0
    in_motion = False
    motion_buffer = []
    low_motion_counter = 0
    detected_frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(raw_fps) if raw_fps and raw_fps > 1 else 30
    start_time = time.monotonic()

    video_placeholder = video_col.empty()
    frame_placeholder = frame_col.empty()
    info_placeholder = frame_col.empty()
    prediction_placeholder = frame_col.empty()

    if live_mode:
        status_placeholder.info(
            f"Live processing started from {source_label} @ ~{fps} FPS "
            f"(max {max_live_seconds}s or {max_live_events} mango detections)"
        )
    else:
        status_placeholder.info(f"Processing {total_frames} frames @ {fps} FPS …")

    while True:
        if live_mode and (time.monotonic() - start_time) >= max_live_seconds:
            status_placeholder.warning("Live session window reached — finalizing results")
            break

        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        roi = extract_roi(frame)
        if roi.size == 0:
            break

        combined_mask, motion_area = process_frame_for_motion(frame, roi, fgbg)
        if live_mode:
            timestamp_ms = int((time.monotonic() - start_time) * 1000)
        else:
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        if frame_idx % 5 == 0:
            display_frame = frame.copy()
            x1, y1, x2, y2 = config.ROI_COORDS
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame_text = (
                f"Live Frame {frame_idx}"
                if live_mode
                else f"Frame {frame_idx}/{total_frames}"
            )
            cv2.putText(display_frame, frame_text,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            video_placeholder.image(
                cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                channels="RGB", width="stretch",
            )

        if not in_motion and motion_area > config.MOTION_AREA_THRESHOLD:
            in_motion = True
            motion_buffer = []
            low_motion_counter = 0
            status_placeholder.warning(f"Motion detected — frame {frame_idx}")

        if in_motion:
            motion_buffer.append((frame.copy(), motion_area, frame_idx, timestamp_ms))
            if motion_area < config.MOTION_AREA_THRESHOLD:
                low_motion_counter += 1
            else:
                low_motion_counter = 0

            if low_motion_counter >= config.MOTION_END_FRAMES:
                in_motion = False
                if motion_buffer:
                    result_data = _process_peak_frame(
                        motion_buffer, frame_col, frame_placeholder,
                        info_placeholder, prediction_placeholder,
                    )
                    detected_frames.append(result_data)
                    if live_mode and len(detected_frames) >= max_live_events:
                        status_placeholder.warning(
                            f"Reached {max_live_events} detections — ending live session"
                        )
                        break
                motion_buffer = []
                low_motion_counter = 0

        if live_mode and len(detected_frames) >= max_live_events:
            break

    # Edge case: motion ongoing at end of video
    if in_motion and motion_buffer:
        result_data = _process_peak_frame(
            motion_buffer, frame_col, frame_placeholder,
            info_placeholder, prediction_placeholder,
        )
        detected_frames.append(result_data)

    cap.release()
    elapsed = int(time.monotonic() - start_time)
    if live_mode:
        status_placeholder.success(
            f"Live processing complete — {len(detected_frames)} mango(es) detected in {elapsed}s"
        )
    else:
        status_placeholder.success(
            f"Processing complete — {len(detected_frames)} mango(es) detected"
        )
    return detected_frames


def _process_peak_frame(motion_buffer, frame_col, frame_ph, info_ph, pred_ph):
    """Extract, classify, segment, and analyse a peak-motion frame."""
    best_frame, best_area, best_idx, best_time = max(motion_buffer, key=lambda x: x[1])
    roi_crop = extract_roi(best_frame)

    roi_rgb = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB)
    frame_ph.image(roi_rgb, caption="Detected Mango", width="stretch")

    info_ph.info("Pre-processing …")
    processed_image = preprocess_image(roi_crop)

    info_ph.info("Classification …")
    prediction = predict_disease(
        st.session_state.model, processed_image, return_probabilities=True
    )

    should_segment = (
        config.ENABLE_SEGMENTATION
        and st.session_state.segmentation_model is not None
    )
    segmentation_result = None
    if should_segment:
        info_ph.info("Segmentation …")
        segmentation_result = segment_disease(
            st.session_state.segmentation_model, roi_crop
        )
        overlay_rgb = cv2.cvtColor(segmentation_result["overlay"], cv2.COLOR_BGR2RGB)
        frame_ph.image(overlay_rgb, caption="Disease Segmentation", width="stretch")

    # ── Disease gate: stop here if diseased ──────────────────────────
    is_classification_diseased = prediction["class_name"] != "Healthy"
    is_segmentation_diseased = (
        segmentation_result is not None
        and segmentation_result["disease_percentage"] > 0
    )
    is_diseased = is_classification_diseased or is_segmentation_diseased

    if is_diseased:
        if is_classification_diseased and is_segmentation_diseased:
            disease_reason = (
                f"Classification: {prediction['class_name']} "
                f"({prediction['confidence']:.0%}) · "
                f"Disease area: {segmentation_result['disease_percentage']:.2f}%"
            )
        elif is_classification_diseased:
            disease_reason = (
                f"Classification: {prediction['class_name']} "
                f"({prediction['confidence']:.0%} confidence)"
            )
        else:
            disease_reason = (
                f"Disease area detected by segmentation: "
                f"{segmentation_result['disease_percentage']:.2f}%"
            )

        pred_ph.error(
            f"🚫 DISEASED — {prediction['class_name']} ({prediction['confidence']:.0%})  \n"
            + (f"Disease area: {segmentation_result['disease_percentage']:.2f}%  \n"
               if segmentation_result else "")
            + "**Cannot be exported.**"
        )

        seg_data = None
        if segmentation_result:
            seg_data = {
                "disease_percentage": segmentation_result["disease_percentage"],
                "overlay": segmentation_result["overlay"],
                "mask": segmentation_result["mask"],
                "segmentation_time": segmentation_result["segmentation_time"],
            }

        result_data = {
            "frame_idx": best_idx,
            "timestamp_ms": best_time,
            "motion_area": best_area,
            "prediction": prediction,
            "original_roi": roi_crop,
            "processed_roi": processed_image,
            "is_diseased": True,
            "disease_reason": disease_reason,
            "defect_analysis": {},
            "segmentation": seg_data,
        }

        try:
            import json
            jsons_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inspection_jsons")
            os.makedirs(jsons_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_filename = f"mango_frame{best_idx}_{ts}.json"
            mango_json = {
                "timestamp": datetime.now().isoformat(),
                "frame_index": best_idx,
                "frame_timestamp_ms": best_time,
                "motion_area": best_area,
                "is_diseased": True,
                "disease_reason": disease_reason,
                "classification": {
                    "class_name": prediction.get("class_name", "N/A"),
                    "confidence": round(prediction.get("confidence", 0), 4),
                    "probabilities": {
                        k: round(v, 4) for k, v in prediction.get("probabilities", {}).items()
                    } if prediction.get("probabilities") else {},
                },
                "defect_analysis": None,
                "segmentation": {
                    "disease_percentage": round(segmentation_result["disease_percentage"], 4),
                    "segmentation_time": round(segmentation_result.get("segmentation_time", 0), 3),
                } if segmentation_result else None,
            }
            with open(os.path.join(jsons_dir, json_filename), "w", encoding="utf-8") as jf:
                json.dump(mango_json, jf, indent=2, ensure_ascii=False)
        except Exception:
            pass

        return result_data

    # ── Healthy: run defect detection ────────────────────────────────
    info_ph.info("Defect detection …")
    hsv_mask = create_mango_hsv_mask(roi_crop)
    mask_224 = cv2.resize(hsv_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
    _, mask_224 = cv2.threshold(mask_224, 127, 255, cv2.THRESH_BINARY)
    defect_result = st.session_state.defect_detector.detect_defects(
        processed_image, mask=mask_224
    )

    defect_vis = st.session_state.defect_detector.visualize_defects(
        processed_image, defect_result
    )

    pred_ph.success(
        f"**{prediction['class_name']}** — {prediction['confidence']:.0%} confidence  \n"
        f"**Quality:** {defect_result.surface_quality_score:.0f}/100 · "
        f"**Impact:** {defect_result.export_grade_impact}"
    )

    result_data = {
        "frame_idx": best_idx,
        "timestamp_ms": best_time,
        "motion_area": best_area,
        "prediction": prediction,
        "original_roi": roi_crop,
        "processed_roi": processed_image,
        "is_diseased": False,
        "defect_analysis": {
            "defect_count": defect_result.defect_count,
            "dark_spot_count": defect_result.dark_spot_count,
            "brown_spot_count": defect_result.brown_spot_count,
            "total_defect_percentage": defect_result.total_defect_percentage,
            "color_uniformity_score": defect_result.color_uniformity_score,
            "surface_quality_score": defect_result.surface_quality_score,
            "export_grade_impact": defect_result.export_grade_impact,
            "defect_visualization": defect_vis,
        },
        "segmentation": None,
    }
    if segmentation_result:
        result_data["segmentation"] = {
            "disease_percentage": segmentation_result["disease_percentage"],
            "overlay": segmentation_result["overlay"],
            "mask": segmentation_result["mask"],
            "segmentation_time": segmentation_result["segmentation_time"],
        }

    # ── Save per-mango JSON to inspection_jsons/ ─────────────────────
    try:
        import json
        jsons_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inspection_jsons")
        os.makedirs(jsons_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"mango_frame{best_idx}_{ts}.json"
        pred = result_data["prediction"]
        da = result_data["defect_analysis"]
        seg = result_data.get("segmentation")
        mango_json = {
            "timestamp": datetime.now().isoformat(),
            "frame_index": best_idx,
            "frame_timestamp_ms": best_time,
            "motion_area": best_area,
            "is_diseased": False,
            "classification": {
                "class_name": pred.get("class_name", "N/A"),
                "confidence": round(pred.get("confidence", 0), 4),
                "probabilities": {
                    k: round(v, 4) for k, v in pred.get("probabilities", {}).items()
                } if pred.get("probabilities") else {},
            },
            "defect_analysis": {
                "surface_quality_score": round(da.get("surface_quality_score", 0), 2),
                "color_uniformity_score": round(da.get("color_uniformity_score", 0), 2),
                "total_defect_percentage": round(da.get("total_defect_percentage", 0), 4),
                "defect_count": da.get("defect_count", 0),
                "dark_spot_count": da.get("dark_spot_count", 0),
                "brown_spot_count": da.get("brown_spot_count", 0),
                "export_grade_impact": da.get("export_grade_impact", "N/A"),
            },
            "segmentation": {
                "disease_percentage": round(seg["disease_percentage"], 4),
                "segmentation_time": round(seg.get("segmentation_time", 0), 3),
            } if seg else None,
        }
        with open(os.path.join(jsons_dir, json_filename), "w", encoding="utf-8") as jf:
            json.dump(mango_json, jf, indent=2, ensure_ascii=False)
    except Exception:
        pass  # Non-critical — don't break the pipeline

    return result_data


# ======================================================================
#   PAGE: Landing / Home
# ======================================================================
def main_page():
    # ── Logos at the top ─────────────────────────────────────────────
    left_col, middle_col, right_col = st.columns([1, 8, 1])

    with left_col:
        st.image("logos/iot_logo.jpeg", width=200)

    with right_col:
        st.image("logos/sal_logo.jpg", width=200)

    # Hero
    st.markdown("""
    <div class="hero-container">
        <h1>ExportEdge AI</h1>
        <p class="subtitle">
            AI-powered mango quality assessment &amp; export advisory system
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    c1, c2, c3, c4 = st.columns(4)
    features = [
        ("🤖", "AI Classification", "Disease classification using AI "),
        ("🔬", "Defect Analysis", "Surface quality, colour uniformity & defect mapping"),
        ("🌍", "Export Advisory", "RAG-powered country recommendations based on regulations"),
        ("💬", "Chat", "Ask questions about export regulations & inspection standards"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], features):
        col.markdown(f"""
        <div class="feature-card">
            <div class="icon">{icon}</div>
            <h3>{title}</h3>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Chat page shortcut
    _, chat_btn_col, _ = st.columns([1, 2, 1])
    with chat_btn_col:
        if st.button("💬 Open Mango Export Chat", use_container_width=True):
            st.session_state.current_page = "chat"
            st.rerun()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Upload section
    col_left, col_mid, col_right = st.columns([1, 2, 1])
    with col_mid:
        st.markdown("### Upload & Start")
        source_type = st.radio(
            "Input source",
            ["Video File", "Camera Feed"],
            index=0,
            horizontal=True,
        )
        st.session_state.source_type = source_type

        video_path = None
        configured_video_source, configured_camera_source = _sources_from_config_py()
        if source_type == "Video File":
            uploaded = st.file_uploader(
                "Upload a video", type=["mp4", "avi", "mov", "mkv"]
            )
            if uploaded:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded.read())
                video_path = tfile.name
                st.success("Video uploaded")
        else:
            video_path = configured_camera_source if configured_camera_source else None
            if video_path:
                st.success("Camera feed will use the configured URL from config.py")
            else:
                st.error("Camera URL not configured in config.py (CAMERA_SOURCE)")

        st.markdown("")
        if st.button("🚀 Start Inspection", type="primary", use_container_width=True):
            st.session_state.last_processing_error = None
            valid_source = False
            if source_type == "Video File":
                valid_source = bool(video_path and os.path.exists(video_path))
            else:
                if video_path and _is_camera_source_reachable(video_path):
                    valid_source = True
                else:
                    st.error("Camera cannot be accessed (You can use Video option)")

            if valid_source:
                load_model()
                _reset_live_runtime_state()
                st.session_state.started = True
                st.session_state.current_page = "processing"
                st.session_state.video_path = video_path
                st.session_state.processing_complete = False
                # Clear stale results from previous run
                st.session_state.results = []
                for k in list(st.session_state.keys()):
                    if k.startswith("rec_") or k in ("report_pdf", "report_name"):
                        del st.session_state[k]
                st.rerun()
            else:
                if source_type == "Video File":
                    st.error("Please select a valid video source")


# ======================================================================
#   PAGE: Processing & Results
# ======================================================================
def processing_page():
    # Top bar
    top_left, top_right = st.columns([6, 1])
    with top_left:
        st.markdown(
            '<h2 style="margin:0;">🥭 Mango Quality Inspection</h2>',
            unsafe_allow_html=True,
        )
    with top_right:
        if st.button("⬅ Home", use_container_width=True):
            _reset_live_runtime_state()
            st.session_state.started = False
            st.session_state.current_page = "home"
            st.session_state.processing_complete = False
            st.session_state.results = []
            st.rerun()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Live processing view ──────────────────────────────────────────
    video_col, frame_col = st.columns([2, 1])
    with video_col:
        st.markdown("##### 📹 Video Feed")
    with frame_col:
        st.markdown("##### 🖼️ Current Detection")

    status_placeholder = st.empty()

    if not st.session_state.processing_complete:
        live_mode = st.session_state.get("source_type") == "Camera Feed"
        source_value = st.session_state.video_path
        source_label = (
            f"camera index {source_value}"
            if isinstance(source_value, int)
            else str(source_value)
        )
        if live_mode:
            c1, c2, _ = st.columns([1, 1, 4])
            with c1:
                if st.button("⏹ Stop Live Session", use_container_width=True):
                    st.session_state.live_stop_requested = True
            with c2:
                st.caption("Live mode active")

            done, live_results = _process_live_video_cycle(
                source_value,
                video_col,
                frame_col,
                status_placeholder,
                source_label=source_label,
                max_live_seconds=st.session_state.get("live_max_seconds", 45),
                max_live_events=st.session_state.get("live_max_events", 6),
            )
            if done:
                st.session_state.results = live_results
                st.session_state.processing_complete = True
            else:
                time.sleep(0.05)
                st.rerun()
        else:
            results = process_video_with_ui(
                source_value,
                video_col,
                frame_col,
                status_placeholder,
                live_mode=False,
                source_label=source_label,
            )
            st.session_state.results = results
            st.session_state.processing_complete = True

    # ── Results (always rendered so buttons stay interactive) ──────────
    if not st.session_state.processing_complete:
        return
    if st.session_state.get("last_processing_error"):
        st.error(st.session_state.last_processing_error)
        return
    results = st.session_state.results
    if not results:
        st.warning("No mangoes detected in the selected source.")
        return

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Summary metrics row ───────────────────────────────────────────
    st.markdown("### 📊 Batch Summary")
    m1, m2, m3, m4 = st.columns(4)
    total = len(results)
    healthy_results = [r for r in results if not r.get("is_diseased")]
    quality_scores = [r.get("defect_analysis", {}).get("surface_quality_score", 0) for r in healthy_results]
    avg_quality = np.mean(quality_scores) if quality_scores else 0
    total_defects = sum(
        r.get("defect_analysis", {}).get("defect_count", 0) for r in healthy_results
    )
    diseased = sum(1 for r in results if r.get("is_diseased"))
    m1.metric("Mangoes Inspected", total)
    m2.metric("Avg. Surface Quality", f"{avg_quality:.0f}/100")
    m3.metric("Total Defects Found", total_defects)
    m4.metric("Diseased Samples", f"{diseased}/{total}")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Per-mango inspection cards ────────────────────────────────────
    st.markdown("### 🔍 Individual Inspections")

    for idx, result in enumerate(results, 1):
        pred = result["prediction"]
        da = result.get("defect_analysis") or {}
        seg = result.get("segmentation")
        impact = da.get("export_grade_impact", "moderate")
        is_diseased = result.get("is_diseased", False)

        if is_diseased:
            expander_title = (
                f"Mango #{idx}  —  🚫 DISEASED ({pred['class_name']})  ·  Cannot be exported"
            )
        else:
            expander_title = (
                f"Mango #{idx}  —  {pred['class_name']}  ·  "
                f"Quality {da.get('surface_quality_score', 0):.0f}/100  ·  "
                f"Impact: {impact.title()}"
            )

        with st.expander(expander_title, expanded=(total == 1)):

            if is_diseased:
                # ── Red banner ────────────────────────────────────
                disease_reason = result.get("disease_reason", "Disease detected")
                st.markdown(f"""
                <div style="background:#450a0a;border:2px solid #ef4444;border-radius:12px;
                            padding:28px 24px;text-align:center;margin:8px 0 16px 0;">
                    <div style="font-size:3rem;margin-bottom:10px;">🚫</div>
                    <h2 style="color:#fca5a5;margin:0 0 10px 0;font-size:1.6rem;">
                        This Mango Cannot Be Exported
                    </h2>
                    <p style="color:#fecaca;font-size:1rem;margin:0 0 14px 0;">
                        It has been identified as <strong>diseased</strong> and does not meet
                        international export quality standards.
                    </p>
                    <span style="color:#f87171;font-size:0.85rem;background:#7f1d1d;
                                 border-radius:6px;padding:6px 14px;display:inline-block;">
                        {disease_reason}
                    </span>
                </div>
                """, unsafe_allow_html=True)

                # Show original + segmentation overlay if available
                img_c1, img_c2 = st.columns(2)
                with img_c1:
                    st.image(
                        cv2.cvtColor(result["original_roi"], cv2.COLOR_BGR2RGB),
                        caption="Detected Mango",
                        width="stretch",
                    )
                if seg and seg.get("overlay") is not None:
                    with img_c2:
                        st.image(
                            cv2.cvtColor(seg["overlay"], cv2.COLOR_BGR2RGB),
                            caption=f"Disease Segmentation ({seg['disease_percentage']:.2f}% coverage)",
                            width="stretch",
                        )

            else:
                # ── Metrics row ──
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("Surface Quality", f"{da.get('surface_quality_score', 0):.0f}/100")
                q2.metric("Colour Uniformity", f"{da.get('color_uniformity_score', 0):.0f}/100")
                q3.metric("Defect Area", f"{da.get('total_defect_percentage', 0):.2f}%")
                if seg:
                    q4.metric("Disease Coverage", f"{seg['disease_percentage']:.2f}%")
                else:
                    q4.metric("Disease Coverage", "N/A")

                # Spot counts + grade
                spot_col, grade_col = st.columns([3, 1])
                with spot_col:
                    st.markdown(
                        f"**Dark spots:** {da.get('dark_spot_count', 0)} · "
                        f"**Brown spots:** {da.get('brown_spot_count', 0)} · "
                        f"**Total defects:** {da.get('defect_count', 0)}"
                    )
                with grade_col:
                    st.markdown(_grade_badge(impact), unsafe_allow_html=True)

                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

                # ── Export Recommendation ─────────────────────────────
                rec_key = f"rec_{idx}"
                if st.button(
                    "🌍 Get Export Recommendation",
                    key=f"btn_rec_{idx}",
                    type="primary",
                ):
                    with st.spinner("Querying export regulations …"):
                        try:
                            seg_pct = seg["disease_percentage"] if seg else 0.0
                            metadata = st.session_state.export_advisor.build_metadata(
                                defect_analysis=da,
                                disease_percentage=seg_pct,
                            )
                            rec = st.session_state.export_advisor.get_recommendation(metadata)

                            # Enrich LLM recommendation with UN Comtrade price estimates.
                            parsed = parse_structured_recommendation(rec.get("answer", ""))
                            recommended = parsed.get("recommended_countries", [])
                            rec["price_estimates"] = st.session_state.price_service.get_prices_for_countries(recommended)

                            st.session_state[rec_key] = rec
                        except Exception as e:
                            st.session_state[rec_key] = {
                                "status": "error",
                                "answer": f"Error: {e}",
                                "sources": [],
                                "price_estimates": [],
                            }

                if rec_key in st.session_state:
                    rec = st.session_state[rec_key]
                    if rec.get("status") == "success":
                        st.markdown("#### 🌍 Export Recommendation")
                        st.markdown(rec["answer"])

                        prices = rec.get("price_estimates", [])
                        if prices:
                            st.markdown("#### 💰 Estimated Export Price (UN Comtrade)")
                            price_rows = []
                            for p in prices:
                                if p.get("status") == "success":
                                    price_rows.append({
                                        "Country": p.get("country", "N/A"),
                                        "Price (USD/kg)": f"{p.get('price_usd_per_kg', 0):.4f}",
                                        "Period": p.get("period", "N/A"),
                                        "Mode": p.get("query_mode", "N/A"),
                                    })
                                else:
                                    price_rows.append({
                                        "Country": p.get("country", "N/A"),
                                        "Price (USD/kg)": "N/A",
                                        "Period": "N/A",
                                        "Mode": p.get("status", "unavailable"),
                                    })
                            st.table(price_rows)

                        if rec.get("sources"):
                            with st.expander("📚 Regulatory Sources"):
                                for s in rec["sources"]:
                                    doc_name = s.get("source", "Unknown")
                                    section = s.get("section", "N/A")
                                    st.write(f"📄 **{doc_name}** — {section}")
                    else:
                        st.error(rec.get("answer", "Unknown error"))

    # ── PDF Report section (only when there are healthy mangoes to report on) ──
    if not healthy_results:
        return

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### 📄 Inspection Report")

    rpt_col1, rpt_col2, rpt_col3 = st.columns([1, 1, 2])
    with rpt_col1:
        generate_report = st.button(
            "📄 Generate PDF Report", type="primary", use_container_width=True
        )
    if generate_report:
        with st.spinner("Generating report …"):
            try:
                recs = {}
                for i in range(1, len(results) + 1):
                    rk = f"rec_{i}"
                    if rk in st.session_state:
                        recs[i] = st.session_state[rk]
                src_value = st.session_state.get("video_path", "")
                if isinstance(src_value, int):
                    video_name = f"camera_{src_value}"
                else:
                    video_name = os.path.basename(src_value)
                pdf_bytes = st.session_state.report_generator.generate(
                    results, recommendations=recs, video_name=video_name,
                )
                st.session_state["report_pdf"] = pdf_bytes
                report_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                report_base = f"mango_inspection_{report_ts}"
                st.session_state["report_name"] = f"{report_base}.pdf"

                # ── Persist report + items to SQLite ────────────
                try:
                    report_id = st.session_state.report_repository.save_report(
                        batch_id=report_base,
                        results=results,
                        recommendations=recs,
                        generated_timestamp=datetime.now().isoformat(),
                    )
                    st.toast(f"DB saved → report_id {report_id}")
                except Exception as db_err:
                    st.warning(f"Database save failed: {db_err}")

                # ── Save structured JSON alongside the PDF ────────
                try:
                    import json
                    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
                    os.makedirs(logs_dir, exist_ok=True)
                    json_path = os.path.join(logs_dir, f"{report_base}.json")

                    json_data = {
                        "report_name": f"{report_base}.pdf",
                        "generated_at": datetime.now().isoformat(),
                        "video_source": video_name,
                        "total_mangoes": len(results),
                        "mangoes": [],
                    }
                    for idx, r in enumerate(results, 1):
                        pred = r.get("prediction", {})
                        da = r.get("defect_analysis", {})
                        seg = r.get("segmentation")
                        mango_entry = {
                            "mango_id": idx,
                            "frame_index": r.get("frame_idx"),
                            "timestamp_ms": r.get("timestamp_ms"),
                            "classification": {
                                "class_name": pred.get("class_name", "N/A"),
                                "confidence": round(pred.get("confidence", 0), 4),
                            },
                            "defect_analysis": {
                                "surface_quality_score": round(da.get("surface_quality_score", 0), 2),
                                "color_uniformity_score": round(da.get("color_uniformity_score", 0), 2),
                                "total_defect_percentage": round(da.get("total_defect_percentage", 0), 4),
                                "defect_count": da.get("defect_count", 0),
                                "dark_spot_count": da.get("dark_spot_count", 0),
                                "brown_spot_count": da.get("brown_spot_count", 0),
                                "export_grade_impact": da.get("export_grade_impact", "N/A"),
                            },
                            "segmentation": {
                                "disease_percentage": round(seg["disease_percentage"], 4),
                                "segmentation_time": round(seg.get("segmentation_time", 0), 3),
                            } if seg else None,
                            "export_recommendation": None,
                        }
                        rec = recs.get(idx)
                        if rec and rec.get("status") == "success":
                            sources_clean = []
                            for s in rec.get("sources", []):
                                sources_clean.append({
                                    "source": s.get("source", ""),
                                    "section": s.get("section", ""),
                                })
                            mango_entry["export_recommendation"] = {
                                "status": "success",
                                "answer": rec.get("answer", ""),
                                "grade": rec.get("grade", ""),
                                "parsed_sections": rec.get("recommendation", {}),
                                "sources": sources_clean,
                                "price_estimates": rec.get("price_estimates", []),
                            }
                        json_data["mangoes"].append(mango_entry)

                    with open(json_path, "w", encoding="utf-8") as jf:
                        json.dump(json_data, jf, indent=2, ensure_ascii=False)
                    st.toast(f"JSON saved → {os.path.basename(json_path)}")
                    
                    # Create embeddings for report so chat can query it
                    try:
                        st.session_state.report_embedding_service.create_report_embeddings(json_data)
                        st.session_state.active_report_json = json_data
                        st.toast("Report embeddings created — chat can now access report context")
                    except Exception as embed_err:
                        st.warning(f"Report embedding failed: {embed_err}")
                except Exception as json_err:
                    st.warning(f"JSON export failed: {json_err}")

                try:
                    log_path = log_inspection(
                        results=results,
                        recommendations=recs,
                        video_name=video_name,
                    )
                    st.toast(f"Inspection logged → {log_path.name}")
                except Exception as log_err:
                    st.warning(f"Logging failed: {log_err}")
            except Exception as e:
                st.error(f"Report generation failed: {e}")

    if "report_pdf" in st.session_state:
        with rpt_col2:
            st.download_button(
                label="⬇️ Download PDF",
                data=st.session_state["report_pdf"],
                file_name=st.session_state.get("report_name", "mango_report.pdf"),
                mime="application/pdf",
                use_container_width=True,
            )


# ======================================================================
#   HELPERS: RAG Chat
# ======================================================================

def _init_chat_rag():
    """Lazy-load the RAG retriever + LLM for the chat page."""
    if st.session_state.chat_rag_ready:
        return True
    try:
        import sys
        from pathlib import Path
        lang_dir = Path(__file__).resolve().parent / "language"
        if str(lang_dir) not in sys.path:
            sys.path.insert(0, str(lang_dir))
        from vector_store_manager import load_vectorstore, create_retriever
        from llm_manager import LLMManager
        from data_config import VECTOR_STORE_PATH, SEARCH_TYPE, RETRIEVAL_K

        vs, _ = load_vectorstore(VECTOR_STORE_PATH)
        st.session_state.chat_retriever = create_retriever(vs, search_type=SEARCH_TYPE, k=RETRIEVAL_K)
        st.session_state.chat_llm = LLMManager()
        st.session_state.chat_rag_ready = True
        return True
    except Exception as e:
        st.error(f"RAG init failed: {e}")
        return False


def _get_rag_context(query: str) -> str:
    """
    Retrieve context for a query, prioritizing active report embeddings.
    Falls back to regulatory documents if no active report.
    """
    context_parts = []
    
    # 1. Check if active report exists and can retrieve
    report_docs = st.session_state.report_embedding_service.retrieve(query, k=3)
    if report_docs:
        context_parts.append("## Active Inspection Report Context:")
        for doc in report_docs:
            mango_id = doc.metadata.get("mango_id", "N/A")
            section = doc.metadata.get("section", "unknown")
            context_parts.append(f"\n[Mango #{mango_id} - {section}]")
            context_parts.append(doc.page_content)
    
    # 2. Fall back to regulatory documents
    if st.session_state.chat_retriever:
        try:
            reg_docs = st.session_state.chat_retriever.invoke(query)
            if reg_docs:
                context_parts.append("\n## Regulatory Documents:")
                for doc in reg_docs:
                    source = doc.metadata.get("source", "Unknown")
                    context_parts.append(f"\n[{source}]")
                    context_parts.append(doc.page_content)
        except Exception:
            pass  # Regulatory retrieval optional
    
    return "\n".join(context_parts) if context_parts else ""


def _build_chat_context_prefix() -> str:
    """If inspection results exist, build a short context summary."""
    results = st.session_state.get("results", [])
    if not results:
        return ""
    lines = ["CURRENT INSPECTION CONTEXT (auto-injected):"]
    for i, r in enumerate(results, 1):
        da = r.get("defect_analysis", {})
        pred = r.get("prediction", {})
        lines.append(
            f"  Mango #{i}: {pred.get('class_name','N/A')}, "
            f"surface={da.get('surface_quality_score',0):.0f}/100, "
            f"uniformity={da.get('color_uniformity_score',0):.0f}/100, "
            f"defects={da.get('total_defect_percentage',0):.2f}%, "
            f"impact={da.get('export_grade_impact','N/A')}"
        )
    return "\n".join(lines) + "\n\n"


# ======================================================================
#   SIDEBAR  (Pipeline info + navigation)
# ======================================================================
def render_sidebar():
    with st.sidebar:
        # ── Header ────────────────────────────────────────────────
        st.markdown(
            '<h2 style="margin-bottom:0;">🥭 ExportEdge AI</h2>',
            unsafe_allow_html=True,
        )
        st.caption("Mango Quality & Export Advisory")
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ── Navigation ───────────────────────────────────────────
        st.markdown("#### Navigation")
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.current_page = "home"
            st.session_state.started = False
            st.rerun()
        if st.button("💬 RAG Chat", use_container_width=True):
            st.session_state.current_page = "chat"
            st.rerun()

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ── Pipeline info ────────────────────────────────────────
        with st.expander("⚙️ Pipeline Info", expanded=False):
            st.markdown(
                f"- **Classification:** `{config.MODEL_PATH.split('/')[-1]}`\n"
                f"- **Input:** {config.INPUT_SHAPE[0]}×{config.INPUT_SHAPE[1]}\n"
                f"- **Segmentation:** DeepLabV3-MobileNet\n"
                f"- **Defect:** OpenCV v3\n"
                f"- **LLM:** DeepSeek R1 (LM Studio)\n"
                f"- **Motion threshold:** {config.MOTION_AREA_THRESHOLD}\n"
                f"- **End-of-motion frames:** {config.MOTION_END_FRAMES}"
            )

        # ── Inspection status ────────────────────────────────────
        results = st.session_state.get("results", [])
        if results:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("#### 📊 Last Inspection")
            st.metric("Mangoes", len(results))
            avg_q = np.mean([
                r.get("defect_analysis", {}).get("surface_quality_score", 0)
                for r in results
            ])
            st.metric("Avg Quality", f"{avg_q:.0f}/100")


# ======================================================================
#   PAGE: RAG Chat
# ======================================================================
def chat_page():
    import html as _html
    import re as _re

    # Top bar
    top_left, top_right = st.columns([6, 1])
    with top_left:
        st.markdown(
            '<h2 style="margin:0;">💬 Mango Export Chat</h2>',
            unsafe_allow_html=True,
        )
    with top_right:
        if st.button("⬅ Home", use_container_width=True, key="chat_home"):
            st.session_state.current_page = "home"
            st.session_state.started = False
            st.rerun()

    st.caption(
        "Ask about export regulations, inspection standards, mango quality — "
    )

    # ── Report status indicator ───────────────────────────────────────
    if st.session_state.active_report_json:
        total_mangoes = st.session_state.active_report_json.get("total_mangoes", 0)
        st.success(
            f"✅ **Active Report Loaded:** {total_mangoes} mango(es) — "
            f"Your questions about this inspection will prioritize the report context."
        )
    else:
        st.info(
            "📝 **No active report:** Run an inspection and generate a report to enable "
            "direct querying about your results. General questions work without a report."
        )

    # ── Inspection context banner ─────────────────────────────────────
    results = st.session_state.get("results", [])
    if results:
        with st.expander("📊 Current inspection context (auto-injected into queries)", expanded=False):
            for i, r in enumerate(results, 1):
                da = r.get("defect_analysis", {})
                pred = r.get("prediction", {})
                st.markdown(
                    f"**Mango #{i}:** {pred.get('class_name', 'N/A')} · "
                    f"Quality {da.get('surface_quality_score', 0):.0f}/100 · "
                    f"Defects {da.get('total_defect_percentage', 0):.2f}% · "
                    f"Impact: {da.get('export_grade_impact', 'N/A').title()}"
                )

    # ── Quick-start prompts ───────────────────────────────────────────
    if not st.session_state.chat_messages:
        st.markdown("##### 💡 Quick Questions")
        quick_prompts = [
            "What are the EU import requirements for mangoes?",
            "How to treat anthracnose before export?",
            "What documents are needed for mango export?",
            "What temperature is required for mango storage?",
        ]
        qp_cols = st.columns(len(quick_prompts))
        for col, qp in zip(qp_cols, quick_prompts):
            with col:
                if st.button(f"💡 {qp}", key=f"qp_{hash(qp)}", use_container_width=True):
                    st.session_state.chat_messages.append({"role": "user", "content": qp})
                    st.rerun()

    # ── Build chat HTML ───────────────────────────────────────────────
    def _md_to_html(text: str) -> str:
        """Minimal markdown → HTML for chat bubbles."""
        text = _html.escape(text)
        # Bold
        text = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        # Italic
        text = _re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
        # Inline code
        text = _re.sub(r"`(.+?)`", r"<code>\1</code>", text)
        # Line breaks
        text = text.replace("\n", "<br>")
        return text

    bubbles_html = ""
    if not st.session_state.chat_messages:
        bubbles_html = (
            '<div class="chat-empty">'
            '<div class="icon">💬</div>'
            '<p>Ask anything about mango export regulations,<br>'
            'quality standards, or your inspection results.</p>'
            '</div>'
        )
    else:
        for msg in st.session_state.chat_messages:
            role = msg["role"]
            content_html = _md_to_html(msg["content"])
            if role == "user":
                bubbles_html += (
                    '<div class="chat-bubble chat-user">'
                    '<div class="chat-label">You</div>'
                    f'{content_html}'
                    '</div>'
                    '<div class="chat-clearfix"></div>'
                )
            else:
                label = "🤖 ExportEdge AI"
                bubbles_html += (
                    f'<div class="chat-bubble chat-assistant">'
                    f'<div class="chat-label">{label}</div>'
                    f'{content_html}'
                    f'</div>'
                    f'<div class="chat-clearfix"></div>'
                )

    # Render chat box
    st.markdown(f'<div class="chat-wrapper">{bubbles_html}</div>', unsafe_allow_html=True)

    # ── Input bar (visually attached to the chat box) ─────────────────
    st.markdown('<div class="chat-input-bar">', unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        input_col, btn_col = st.columns([9, 1])
        with input_col:
            user_input = st.text_input(
                "Message",
                placeholder="Ask about mango export regulations, quality standards, or your inspection results...",
                label_visibility="collapsed",
                key="chat_text_input",
            )
        with btn_col:
            submitted = st.form_submit_button("➤", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if submitted and user_input and user_input.strip():
        st.session_state.chat_messages.append({"role": "user", "content": user_input.strip()})

    # ── Sources expander ──────────────────────────────────────────────
    source_msgs = [
        msg for msg in st.session_state.chat_messages
        if msg.get("sources") and msg["role"] == "assistant"
    ]
    if source_msgs:
        last_src = source_msgs[-1]
        with st.expander("📚 Sources from latest response", expanded=False):
            for s in last_src["sources"]:
                doc_name = s.get("source", "Unknown")
                section = s.get("section", "N/A")
                st.write(f"📄 **{doc_name}** — {section}")

    # ── Process latest user message ───────────────────────────────────
    if (
        st.session_state.chat_messages
        and st.session_state.chat_messages[-1]["role"] == "user"
    ):
        user_query = st.session_state.chat_messages[-1]["content"]
        with st.spinner("🔍 Retrieving context & generating answer — this may take a couple of minutes..."):
            if _init_chat_rag():
                # Build RAG context, prioritizing active report embeddings
                rag_context = _get_rag_context(user_query)
                
                # Build final query with context
                context_prefix = _build_chat_context_prefix()
                full_query = (context_prefix + rag_context + "\n\nUser query: " + user_query 
                             if rag_context 
                             else (context_prefix + user_query if context_prefix else user_query))

                # Create full prompt for LLM
                prompt = (
                    "You are an expert assistant for mango inspection and export regulations. "
                    "Use the provided context to answer questions accurately and comprehensively.\n\n"
                    "CONTEXT:\n" + full_query + "\n\n"
                    "INSTRUCTIONS:\n"
                    "1. Answer based primarily on the provided context\n"
                    "2. Be specific and cite relevant sections when possible\n"
                    "3. If the context doesn't contain enough information, say so clearly\n"
                    "4. Focus on practical, actionable information\n"
                    "5. Use bullet points or numbered lists for clarity when appropriate\n\n"
                    "ANSWER:"
                )
                
                # Generate answer
                answer = st.session_state.chat_llm.query_llm(prompt)
                answer = answer if answer else "Sorry, I couldn't generate a response."
                
                # Remove thinking blocks if present
                import re
                answer = re.sub(r"<think>[\s\S]*?</think>", "", answer).strip()

                # Extract sources from active report if available
                sources = []
                if st.session_state.report_embedding_service.retriever:
                    try:
                        docs = st.session_state.report_embedding_service.retrieve(user_query, k=2)
                        for doc in docs:
                            sources.append({
                                "source": doc.metadata.get("source", "Active Report"),
                                "section": doc.metadata.get("section", "N/A"),
                            })
                    except Exception:
                        pass
                
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
            else:
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": "⚠️ Could not connect to the RAG pipeline. Make sure LM Studio is running.",
                })
        st.rerun()

    # ── Footer controls ───────────────────────────────────────────────
    if st.session_state.chat_messages:
        f1, f2, _ = st.columns([1, 1, 4])
        with f1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_messages = []
                st.rerun()
        with f2:
            if results:
                if st.button("📊 Back to Results", use_container_width=True):
                    st.session_state.current_page = "processing"
                    st.rerun()


# ======================================================================
#   ENTRY POINT
# ======================================================================
def main():
    render_sidebar()
    page = st.session_state.current_page
    if page == "chat":
        chat_page()
    elif st.session_state.started or page == "processing":
        processing_page()
    else:
        main_page()


if __name__ == "__main__":
    main()
