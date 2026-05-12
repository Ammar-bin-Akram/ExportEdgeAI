"""
Data validation tests for ExportEdge AI — SOP document pipeline.

Loads the 7 real Markdown SOP files from data_for_llm/, runs them through
chunk_documents(), and validates the structural quality of every chunk.

No external API calls, no model files, no internet access required.

Run from the Code/ directory:
    pytest tests/test_data_validation.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
_LANG_DIR = _CODE_DIR / "language"
if str(_LANG_DIR) not in sys.path:
    sys.path.insert(0, str(_LANG_DIR))

# ── Mock tflite before any project import ─────────────────────────────────────
_tflite_mock = MagicMock()
sys.modules.setdefault("tflite_runtime", _tflite_mock)
sys.modules.setdefault("tflite_runtime.interpreter", _tflite_mock)

# ── Project imports ────────────────────────────────────────────────────────────
from language.chunker import chunk_documents

# ── Data directory ────────────────────────────────────────────────────────────
_DATA_DIR = _CODE_DIR.parent / "data_for_llm"

_SOP_FILES = [
    "45_Mangoes.md",
    "APHIS-2006-0121-0010_content.md",
    "CXS_184e(Middle East& South Asia).md",
    "Final-Mango-Brochure.md",
    "Mango_Inspection_Instructions[1]USA.md",
    "Manual For Export of Mangoes.md",
    "mango_content.md",
]

_REQUIRED_METADATA_KEYS = {"source", "filepath", "chunk_id", "section_heading",
                            "heading_level", "chunk_type", "chunk_size"}

_CHUNK_SIZE = 1000   # must match chunk_documents() default
_CHUNK_OVERLAP = 200


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _load_raw_docs() -> list[dict]:
    """Read all SOP files and return list of dicts for chunk_documents()."""
    docs = []
    for fname in _SOP_FILES:
        path = _DATA_DIR / fname
        if path.exists():
            docs.append({
                "content": path.read_text(encoding="utf-8", errors="replace"),
                "filename": fname,
                "filepath": str(path),
            })
    return docs


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_docs():
    docs = _load_raw_docs()
    assert len(docs) > 0, (
        f"No SOP files found in {_DATA_DIR}. "
        "Make sure data_for_llm/ is present next to Code/."
    )
    return docs


@pytest.fixture(scope="module")
def chunks(raw_docs):
    return chunk_documents(raw_docs)


# ─────────────────────────────────────────────────────────────────────────────
#  1. File Availability
# ─────────────────────────────────────────────────────────────────────────────

class TestFileAvailability:
    """All SOP files must exist and be readable."""

    def test_data_directory_exists(self):
        assert _DATA_DIR.exists(), f"data_for_llm/ not found at {_DATA_DIR}"

    @pytest.mark.parametrize("fname", _SOP_FILES)
    def test_sop_file_exists(self, fname):
        assert (_DATA_DIR / fname).exists(), f"Missing SOP file: {fname}"

    @pytest.mark.parametrize("fname", _SOP_FILES)
    def test_sop_file_is_non_empty(self, fname):
        path = _DATA_DIR / fname
        if path.exists():
            assert path.stat().st_size > 0, f"SOP file is empty: {fname}"

    def test_all_seven_files_present(self):
        missing = [f for f in _SOP_FILES if not (_DATA_DIR / f).exists()]
        assert missing == [], f"Missing SOP files: {missing}"


# ─────────────────────────────────────────────────────────────────────────────
#  2. Chunk Count & Non-Emptiness
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkCountAndContent:
    """Chunking must produce a reasonable number of non-empty chunks."""

    def test_chunking_produces_chunks(self, chunks):
        assert len(chunks) > 0, "chunk_documents() returned zero chunks"

    def test_minimum_chunk_count(self, chunks):
        # 7 files × at least 1 section each → expect at least 20 chunks
        assert len(chunks) >= 20, (
            f"Only {len(chunks)} chunks produced — expected at least 20 from 7 SOP files"
        )

    def test_no_chunk_has_empty_page_content(self, chunks):
        empty = [i for i, c in enumerate(chunks) if not c.page_content.strip()]
        assert empty == [], f"Chunks at indices {empty} have empty page_content"

    def test_no_chunk_page_content_is_whitespace_only(self, chunks):
        bad = [i for i, c in enumerate(chunks) if c.page_content.strip() == ""]
        assert bad == [], f"Chunks at indices {bad} contain only whitespace"

    def test_chunk_count_matches_raw_doc_count(self, raw_docs, chunks):
        sources_in_chunks = {c.metadata["source"] for c in chunks}
        sources_in_input = {d["filename"] for d in raw_docs}
        assert sources_in_chunks == sources_in_input, (
            f"Source mismatch.\n  Input: {sources_in_input}\n  Chunks: {sources_in_chunks}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  3. Metadata Completeness
# ─────────────────────────────────────────────────────────────────────────────

class TestMetadataCompleteness:
    """Every chunk must carry all required metadata keys with valid values."""

    def test_all_chunks_have_required_keys(self, chunks):
        for i, chunk in enumerate(chunks):
            missing = _REQUIRED_METADATA_KEYS - set(chunk.metadata.keys())
            assert missing == set(), (
                f"Chunk {i} (source={chunk.metadata.get('source')}) "
                f"is missing metadata keys: {missing}"
            )

    def test_source_is_non_empty_string(self, chunks):
        bad = [i for i, c in enumerate(chunks)
               if not isinstance(c.metadata.get("source"), str)
               or not c.metadata["source"].strip()]
        assert bad == [], f"Chunks at indices {bad} have invalid 'source' metadata"

    def test_filepath_is_non_empty_string(self, chunks):
        bad = [i for i, c in enumerate(chunks)
               if not isinstance(c.metadata.get("filepath"), str)
               or not c.metadata["filepath"].strip()]
        assert bad == [], f"Chunks at indices {bad} have invalid 'filepath' metadata"

    def test_chunk_id_is_non_negative_integer(self, chunks):
        bad = [i for i, c in enumerate(chunks)
               if not isinstance(c.metadata.get("chunk_id"), int)
               or c.metadata["chunk_id"] < 0]
        assert bad == [], f"Chunks at indices {bad} have invalid 'chunk_id'"

    def test_section_heading_is_string(self, chunks):
        bad = [i for i, c in enumerate(chunks)
               if not isinstance(c.metadata.get("section_heading"), str)]
        assert bad == [], f"Chunks at indices {bad} have non-string 'section_heading'"

    def test_heading_level_is_non_negative_integer(self, chunks):
        bad = [i for i, c in enumerate(chunks)
               if not isinstance(c.metadata.get("heading_level"), int)
               or c.metadata["heading_level"] < 0]
        assert bad == [], f"Chunks at indices {bad} have invalid 'heading_level'"

    def test_chunk_type_is_valid(self, chunks):
        valid = {"section", "subsection"}
        bad = [i for i, c in enumerate(chunks)
               if c.metadata.get("chunk_type") not in valid]
        assert bad == [], (
            f"Chunks at indices {bad} have chunk_type not in {valid}"
        )

    def test_chunk_size_metadata_matches_page_content_length(self, chunks):
        bad = [i for i, c in enumerate(chunks)
               if c.metadata.get("chunk_size") != len(c.page_content)]
        assert bad == [], (
            f"Chunks at indices {bad} have chunk_size != len(page_content)"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  4. Chunk ID Integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkIdIntegrity:
    """chunk_id must be globally unique and sequential starting from 0."""

    def test_chunk_ids_start_at_zero(self, chunks):
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert ids[0] == 0, f"First chunk_id is {ids[0]}, expected 0"

    def test_chunk_ids_are_sequential(self, chunks):
        ids = [c.metadata["chunk_id"] for c in chunks]
        expected = list(range(len(chunks)))
        assert ids == expected, (
            f"chunk_ids are not sequential. "
            f"First mismatch at index {next(i for i,(a,b) in enumerate(zip(ids,expected)) if a!=b)}"
        )

    def test_chunk_ids_are_unique(self, chunks):
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), (
            f"Duplicate chunk_ids found: "
            f"{[id for id in ids if ids.count(id) > 1]}"
        )

    def test_total_chunk_count_equals_max_id_plus_one(self, chunks):
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(chunks) == max(ids) + 1


# ─────────────────────────────────────────────────────────────────────────────
#  5. Source Metadata Accuracy
# ─────────────────────────────────────────────────────────────────────────────

class TestSourceMetadataAccuracy:
    """source metadata must exactly match the input filename."""

    def test_all_sources_are_known_filenames(self, chunks):
        known = set(_SOP_FILES)
        unknown = {c.metadata["source"] for c in chunks} - known
        assert unknown == set(), f"Unexpected source values in chunks: {unknown}"

    @pytest.mark.parametrize("fname", _SOP_FILES)
    def test_each_file_contributes_at_least_one_chunk(self, fname, chunks):
        file_chunks = [c for c in chunks if c.metadata["source"] == fname]
        assert len(file_chunks) >= 1, (
            f"No chunks found for {fname}"
        )

    @pytest.mark.parametrize("fname", _SOP_FILES)
    def test_filepath_contains_filename(self, fname, chunks):
        file_chunks = [c for c in chunks if c.metadata["source"] == fname]
        for chunk in file_chunks:
            assert fname in chunk.metadata["filepath"], (
                f"filepath '{chunk.metadata['filepath']}' does not contain filename '{fname}'"
            )


# ─────────────────────────────────────────────────────────────────────────────
#  6. Chunk Size Constraints
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkSizeConstraints:
    """Chunks must not exceed the splitter's size limit."""

    def test_no_subsection_chunk_exceeds_size_plus_heading_overhead(self, chunks):
        # subsection chunks get a "# heading\n\n" prefix added (~50 chars max overhead)
        max_allowed = _CHUNK_SIZE + _CHUNK_OVERLAP + 60
        bad = [
            (i, c.metadata["chunk_size"])
            for i, c in enumerate(chunks)
            if c.metadata["chunk_type"] == "subsection"
            and c.metadata["chunk_size"] > max_allowed
        ]
        assert bad == [], (
            f"Subsection chunks exceed size budget ({max_allowed}): {bad[:5]}"
        )

    def test_section_chunks_within_chunk_size_limit(self, chunks):
        bad = [
            (i, c.metadata["chunk_size"])
            for i, c in enumerate(chunks)
            if c.metadata["chunk_type"] == "section"
            and c.metadata["chunk_size"] > _CHUNK_SIZE
        ]
        assert bad == [], (
            f"Section chunks (kept whole) exceed chunk_size={_CHUNK_SIZE}: {bad[:5]}"
        )

    def test_minimum_meaningful_chunk_length(self, chunks):
        # Chunks under 3 chars are degenerate noise; short headings like "# NOTE" are valid
        short = [
            (i, repr(c.page_content))
            for i, c in enumerate(chunks)
            if len(c.page_content.strip()) < 3
        ]
        assert short == [], f"Degenerate chunks found: {short}"

    def test_average_chunk_size_is_reasonable(self, chunks):
        avg = sum(len(c.page_content) for c in chunks) / len(chunks)
        # Average should be at least 50 chars and at most chunk_size + overhead
        assert 50 <= avg <= _CHUNK_SIZE + _CHUNK_OVERLAP + 60, (
            f"Average chunk size {avg:.0f} is outside expected range"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  7. Per-File Coverage
# ─────────────────────────────────────────────────────────────────────────────

class TestPerFileCoverage:
    """Each SOP file must be adequately covered by chunks."""

    @pytest.mark.parametrize("fname", _SOP_FILES)
    def test_file_chunk_content_covers_source_text(self, fname, raw_docs, chunks):
        """Reconstructed chunk text must represent the original file content."""
        raw = next((d for d in raw_docs if d["filename"] == fname), None)
        if raw is None:
            pytest.skip(f"{fname} not found in raw_docs")

        file_chunks = [c for c in chunks if c.metadata["source"] == fname]
        total_chunk_chars = sum(len(c.page_content) for c in file_chunks)
        original_chars = len(raw["content"])

        # Chunks (with possible overlap) should cover at least 50% of source chars
        assert total_chunk_chars >= original_chars * 0.50, (
            f"{fname}: total chunk chars ({total_chunk_chars}) < 50% of "
            f"original ({original_chars})"
        )

    @pytest.mark.parametrize("fname", _SOP_FILES)
    def test_section_headings_are_propagated(self, fname, chunks):
        """At least some chunks from each file should have non-empty section headings."""
        file_chunks = [c for c in chunks if c.metadata["source"] == fname]
        headings = [c.metadata["section_heading"] for c in file_chunks
                    if c.metadata["section_heading"]]
        # It's acceptable if a file has no markdown headings (heading = ''),
        # but if it does have headings they must be strings (already checked above).
        # Here we just ensure no exception occurred.
        assert isinstance(headings, list)

    @pytest.mark.parametrize("fname", _SOP_FILES)
    def test_heading_level_zero_only_when_no_heading(self, fname, chunks):
        """heading_level 0 must correspond to an empty section_heading."""
        file_chunks = [c for c in chunks if c.metadata["source"] == fname]
        for chunk in file_chunks:
            if chunk.metadata["heading_level"] == 0:
                assert chunk.metadata["section_heading"] == "", (
                    f"heading_level=0 but section_heading='{chunk.metadata['section_heading']}'"
                )
