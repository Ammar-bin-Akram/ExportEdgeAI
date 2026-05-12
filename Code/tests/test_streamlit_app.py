"""
Streamlit AppTest suite for ExportEdge AI.

Tests UI structure, session state, navigation, and quick-prompt behaviour
without real model inference or external API calls.

All heavy / unavailable modules (TFLite, PyTorch, HuggingFace, OpenAI) are
mocked at the sys.modules level before AppTest loads the script so that every
import inside the spinner block succeeds and model-loading try/except blocks
pass silently.

Run from the Code/ directory:
    pytest tests/test_streamlit_app.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

# ── Path setup (mirrors conftest.py) ──────────────────────────────────────────
_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

# language/ subpackage uses bare `from data_config import …`
_LANG_DIR = _CODE_DIR / "language"
if str(_LANG_DIR) not in sys.path:
    sys.path.insert(0, str(_LANG_DIR))

# ── Mock every heavy / unavailable module BEFORE AppTest loads the script ─────

# 1. TFLite / TensorFlow — prevents the broken protobuf import chain
_tflite_mock = MagicMock()
sys.modules.setdefault("tflite_runtime", _tflite_mock)
sys.modules.setdefault("tflite_runtime.interpreter", _tflite_mock)

# 2. model_utils — avoids loading the .tflite file
_model_utils_mock = MagicMock()
_model_utils_mock.load_model_weights.return_value = (
    MagicMock(), [{"index": 0}], [{"index": 0}]
)
_model_utils_mock.predict_disease.return_value = {
    "class_name": "Healthy",
    "confidence": 0.95,
    "probabilities": {
        "Healthy": 0.95, "Alternaria": 0.02, "Anthracnose": 0.01,
        "Black Mould Rot": 0.01, "Stem end Rot": 0.01,
    },
}
sys.modules["model_utils"] = _model_utils_mock

# 3. segmentation_utils — avoids loading the .pth file
_seg_utils_mock = MagicMock()
_seg_utils_mock.load_segmentation_model.return_value = None
sys.modules["segmentation_utils"] = _seg_utils_mock

# 4. video_utils — no real video needed
_video_utils_mock = MagicMock()
_video_utils_mock.create_background_subtractor.return_value = MagicMock()
_video_utils_mock.extract_roi.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
_video_utils_mock.process_frame_for_motion.return_value = (
    np.zeros((224, 224), dtype=np.uint8), 0
)
sys.modules["video_utils"] = _video_utils_mock

# 5. preprocessing
_preprocessing_mock = MagicMock()
_preprocessing_mock.preprocess_image.return_value = np.zeros((150, 150, 3), dtype=np.uint8)
sys.modules["preprocessing"] = _preprocessing_mock

# 6. language.report_embeddings — avoid loading HuggingFace embeddings model
_report_emb_mock = MagicMock()
_report_emb_mock.ReportEmbeddingService.return_value.retrieve.return_value = []
_report_emb_mock.ReportEmbeddingService.return_value.retriever = None
sys.modules["language.report_embeddings"] = _report_emb_mock

# 7. language.comtrade_price_service — no UN Comtrade API calls
sys.modules["language.comtrade_price_service"] = MagicMock()

# 8. vision.export_report_repository — avoid SQLite side-effects
sys.modules["vision.export_report_repository"] = MagicMock()

# 9. vision.inspection_logger — avoid file system writes
sys.modules["vision.inspection_logger"] = MagicMock()

# 10. RAG internals used by _init_chat_rag() — called during chat queries
#     vector_store_manager (bare import from language/ subpackage)
_mock_retriever = MagicMock()
_mock_retriever.invoke.return_value = []          # no documents returned
_vs_manager_mock = MagicMock()
_vs_manager_mock.load_vectorstore.return_value = (MagicMock(), MagicMock())
_vs_manager_mock.create_retriever.return_value = _mock_retriever
sys.modules["vector_store_manager"] = _vs_manager_mock

#     llm_manager (bare import from language/ subpackage)
_llm_manager_mod_mock = MagicMock()
_llm_manager_mod_mock.LLMManager.return_value.query_llm.return_value = (
    "Grade A mangoes are suitable for EU and Japan markets."
)
sys.modules["llm_manager"] = _llm_manager_mod_mock

# ── Import AppTest only after all mocks are in place ──────────────────────────
from streamlit.testing.v1 import AppTest  # noqa: E402

_APP_PATH = str(_CODE_DIR / "streamlit_app.py")
_TIMEOUT = 30  # seconds per AppTest run


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fresh() -> AppTest:
    """Return a freshly rendered AppTest instance on the home page."""
    return AppTest.from_file(_APP_PATH, default_timeout=_TIMEOUT).run()


def _find_button(at: AppTest, label_substr: str):
    """Return the first button whose label contains label_substr, or None."""
    return next((b for b in at.button if label_substr in b.label), None)


def _button_exists(at: AppTest, label_substr: str) -> bool:
    return _find_button(at, label_substr) is not None


def _on_chat_page(at: AppTest | None = None) -> AppTest:
    """Return an AppTest instance rendered on the chat page."""
    at = at or _fresh()
    at.session_state["current_page"] = "chat"
    return at.run()


# ─────────────────────────────────────────────────────────────────────────────
#  1. Session State Initialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionStateInitialisation:
    """All required session-state keys must be present with correct defaults."""

    def test_no_exceptions_on_first_render(self):
        assert not _fresh().exception

    def test_current_page_defaults_to_home(self):
        assert _fresh().session_state["current_page"] == "home"

    def test_started_defaults_to_false(self):
        assert _fresh().session_state["started"] is False

    def test_results_defaults_to_empty_list(self):
        assert _fresh().session_state["results"] == []

    def test_chat_messages_defaults_to_empty_list(self):
        assert _fresh().session_state["chat_messages"] == []

    def test_processing_complete_defaults_to_false(self):
        assert _fresh().session_state["processing_complete"] is False

    def test_source_type_defaults_to_video_file(self):
        assert _fresh().session_state["source_type"] == "Video File"

    def test_chat_rag_ready_defaults_to_false(self):
        assert _fresh().session_state["chat_rag_ready"] is False

    def test_active_report_json_defaults_to_none(self):
        assert _fresh().session_state["active_report_json"] is None


# ─────────────────────────────────────────────────────────────────────────────
#  2. Home Page Widgets
# ─────────────────────────────────────────────────────────────────────────────

class TestHomePageWidgets:
    """All expected controls must render on the home page."""

    def test_start_inspection_button_exists(self):
        assert _button_exists(_fresh(), "Start Inspection")

    def test_open_chat_shortcut_button_exists(self):
        assert _button_exists(_fresh(), "Open Mango Export Chat")

    def test_source_type_radio_exists(self):
        assert len(_fresh().radio) > 0

    def test_source_type_radio_has_video_file_option(self):
        assert "Video File" in _fresh().radio[0].options

    def test_source_type_radio_has_camera_feed_option(self):
        assert "Camera Feed" in _fresh().radio[0].options

    def test_source_type_radio_defaults_to_video_file(self):
        assert _fresh().radio[0].value == "Video File"

    def test_use_default_video_checkbox_exists(self):
        assert len(_fresh().checkbox) > 0

    def test_use_default_video_checkbox_is_checked_by_default(self):
        assert _fresh().checkbox[0].value is True

    def test_unchecking_default_video_updates_checkbox_state(self):
        at = _fresh()
        at.checkbox[0].uncheck().run()
        assert at.checkbox[0].value is False


# ─────────────────────────────────────────────────────────────────────────────
#  3. Home Page Navigation
# ─────────────────────────────────────────────────────────────────────────────

class TestHomePageNavigation:
    """Button clicks on the home page must trigger the correct page transitions."""

    def test_chat_shortcut_button_navigates_to_chat_page(self):
        at = _fresh()
        _find_button(at, "Open Mango Export Chat").click().run()
        assert at.session_state["current_page"] == "chat"

    def test_start_inspection_without_video_does_not_navigate(self):
        """With no valid video source, clicking Start should not leave home."""
        at = _fresh()
        at.checkbox[0].uncheck().run()          # hide default video → no path
        _find_button(at, "Start Inspection").click().run()
        assert at.session_state["current_page"] == "home"
        assert at.session_state["started"] is False

    def test_start_inspection_without_video_stays_on_home(self):
        """Verify started flag remains False when no video is selected."""
        at = _fresh()
        at.checkbox[0].uncheck().run()
        _find_button(at, "Start Inspection").click().run()
        assert at.session_state["started"] is False


# ─────────────────────────────────────────────────────────────────────────────
#  4. Sidebar Widgets
# ─────────────────────────────────────────────────────────────────────────────

class TestSidebarWidgets:
    """Sidebar navigation buttons must render and function correctly."""

    def test_sidebar_home_button_exists(self):
        assert _button_exists(_fresh(), "🏠 Home")

    def test_sidebar_rag_chat_button_exists(self):
        assert _button_exists(_fresh(), "RAG Chat")

    def test_sidebar_rag_chat_button_navigates_to_chat(self):
        at = _fresh()
        _find_button(at, "RAG Chat").click().run()
        assert at.session_state["current_page"] == "chat"

    def test_sidebar_home_button_navigates_to_home(self):
        """Start on chat page, click sidebar Home, land on home."""
        at = _on_chat_page()
        _find_button(at, "🏠 Home").click().run()
        assert at.session_state["current_page"] == "home"
        assert at.session_state["started"] is False


# ─────────────────────────────────────────────────────────────────────────────
#  5. Chat Page Structure
# ─────────────────────────────────────────────────────────────────────────────

class TestChatPage:
    """Chat page must render required widgets and handle quick-prompt clicks."""

    def test_chat_page_renders_without_exceptions(self):
        assert not _on_chat_page().exception

    def test_chat_page_has_text_input(self):
        assert len(_on_chat_page().text_input) > 0

    def test_chat_text_input_has_placeholder(self):
        at = _on_chat_page()
        assert at.text_input[0].placeholder != ""

    def test_chat_page_has_four_quick_prompt_buttons(self):
        at = _on_chat_page()
        quick_btns = [b for b in at.button if "💡" in b.label]
        assert len(quick_btns) == 4

    def test_quick_prompt_labels_are_non_empty(self):
        at = _on_chat_page()
        for btn in (b for b in at.button if "💡" in b.label):
            assert len(btn.label.replace("💡", "").strip()) > 0

    def test_quick_prompt_click_adds_user_message(self):
        at = _on_chat_page()
        [b for b in at.button if "💡" in b.label][0].click().run()
        user_msgs = [m for m in at.session_state["chat_messages"] if m["role"] == "user"]
        assert len(user_msgs) == 1

    def test_quick_prompt_user_message_content_is_non_empty(self):
        at = _on_chat_page()
        [b for b in at.button if "💡" in b.label][0].click().run()
        user_msgs = [m for m in at.session_state["chat_messages"] if m["role"] == "user"]
        assert len(user_msgs[0]["content"]) > 0

    def test_quick_prompt_triggers_assistant_response(self):
        """After a user message, the mocked LLM should add an assistant reply."""
        at = _on_chat_page()
        [b for b in at.button if "💡" in b.label][0].click().run()
        asst_msgs = [m for m in at.session_state["chat_messages"] if m["role"] == "assistant"]
        assert len(asst_msgs) == 1

    def test_assistant_response_content_is_non_empty(self):
        at = _on_chat_page()
        [b for b in at.button if "💡" in b.label][0].click().run()
        asst_msgs = [m for m in at.session_state["chat_messages"] if m["role"] == "assistant"]
        assert len(asst_msgs[0]["content"]) > 0

    def test_clear_chat_button_appears_after_messages(self):
        at = _on_chat_page()
        [b for b in at.button if "💡" in b.label][0].click().run()
        assert _button_exists(at, "Clear Chat")

    def test_clear_chat_button_empties_messages(self):
        at = _on_chat_page()
        [b for b in at.button if "💡" in b.label][0].click().run()
        _find_button(at, "Clear Chat").click().run()
        assert at.session_state["chat_messages"] == []

    def test_quick_prompts_hidden_after_first_message(self):
        """Quick-start prompts should disappear once chat_messages is non-empty."""
        at = _on_chat_page()
        [b for b in at.button if "💡" in b.label][0].click().run()
        # After messages exist, quick prompt buttons should no longer render
        quick_btns = [b for b in at.button if "💡" in b.label]
        assert len(quick_btns) == 0

    def test_chat_home_button_navigates_back(self):
        at = _on_chat_page()
        _find_button(at, "⬅ Home").click().run()
        assert at.session_state["current_page"] == "home"

    def test_no_active_report_info_banner_shown(self):
        """Without a report, the info state must be correctly reflected."""
        at = _on_chat_page()
        assert at.session_state["active_report_json"] is None
        assert not at.exception


# ─────────────────────────────────────────────────────────────────────────────
#  6. Input Source Radio
# ─────────────────────────────────────────────────────────────────────────────

class TestInputSourceRadio:
    """Source-type radio must update session state correctly on change."""

    def test_defaults_to_video_file(self):
        assert _fresh().radio[0].value == "Video File"

    def test_switching_to_camera_feed_updates_session_state(self):
        at = _fresh()
        at.radio[0].set_value("Camera Feed").run()
        assert at.session_state["source_type"] == "Camera Feed"

    def test_switching_back_to_video_file_updates_session_state(self):
        at = _fresh()
        at.radio[0].set_value("Camera Feed").run()
        at.radio[0].set_value("Video File").run()
        assert at.session_state["source_type"] == "Video File"

    def test_camera_feed_hides_default_video_checkbox(self):
        """Switching to Camera Feed should not show the video file checkbox."""
        at = _fresh()
        at.radio[0].set_value("Camera Feed").run()
        # Checkbox only appears for Video File source
        assert len(at.checkbox) == 0
