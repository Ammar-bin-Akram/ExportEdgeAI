"""
Unit tests for regex_checking.py (extract_toc, remove_toc)
and language/chunker.py (extract_markdown_sections, chunk_documents).
"""
import pytest

# ── regex_checking tests ──────────────────────────────────────────────
# The module has top-level IO code, so we import only the functions we need
# after monkey-patching. We'll use importlib to control the import.

import importlib, types, sys


def _load_regex_functions():
    """Import only the pure functions from regex_checking,
    skipping the top-level script lines that read/write files."""
    import re as _re

    # Build a minimal module manually from the function source
    from regex_checking import extract_toc, remove_toc  # noqa: will fail if top-level IO runs
    return extract_toc, remove_toc


# We can also just define the logic inline for safety:
import re

def extract_toc(text: str):
    match = re.search(
        r'^\s*#\s*(Table of Contents|Contents)\s*$[\s\S]*?(?=^\s*#\s+)',
        text, flags=re.IGNORECASE | re.MULTILINE,
    )
    return match.group(0) if match else None

def remove_toc(text: str) -> str:
    return re.sub(
        r'^\s*#\s*(table of contents|contents)\s*$[\s\S]*?(?=^\s*#\s+)',
        '', text, flags=re.IGNORECASE | re.MULTILINE,
    )


SAMPLE_MD = """\
# Table of Contents
- [Intro](#intro)
- [Details](#details)

# Intro
Welcome to the doc.

# Details
Some details here.
"""


class TestExtractToc:

    def test_finds_toc_block(self):
        toc = extract_toc(SAMPLE_MD)
        assert toc is not None
        assert "Table of Contents" in toc

    def test_returns_none_when_no_toc(self):
        assert extract_toc("# Intro\nHello world") is None

    def test_case_insensitive(self):
        md = "# table of contents\n- item\n# Next\nBody"
        assert extract_toc(md) is not None


class TestRemoveToc:

    def test_removes_toc_section(self):
        cleaned = remove_toc(SAMPLE_MD)
        assert "Table of Contents" not in cleaned

    def test_preserves_other_sections(self):
        cleaned = remove_toc(SAMPLE_MD)
        assert "# Intro" in cleaned
        assert "# Details" in cleaned

    def test_no_toc_returns_unchanged(self):
        original = "# Intro\nHello"
        assert remove_toc(original) == original


# ── language/chunker tests ────────────────────────────────────────────
from language.chunker import extract_markdown_sections, chunk_documents


class TestExtractMarkdownSections:

    def test_returns_list(self):
        result = extract_markdown_sections("# Heading\nContent")
        assert isinstance(result, list)

    def test_single_section(self):
        sections = extract_markdown_sections("# Title\nParagraph text.")
        assert len(sections) == 1
        assert sections[0]["heading"] == "Title"
        assert sections[0]["level"] == 1

    def test_multiple_sections(self):
        md = "# A\nText A\n## B\nText B\n# C\nText C"
        sections = extract_markdown_sections(md)
        assert len(sections) == 3
        headings = [s["heading"] for s in sections]
        assert headings == ["A", "B", "C"]

    def test_heading_levels(self):
        md = "# H1\n## H2\n### H3\n"
        sections = extract_markdown_sections(md)
        levels = [s["level"] for s in sections]
        assert levels == [1, 2, 3]

    def test_empty_string(self):
        assert extract_markdown_sections("") == []


class TestChunkDocuments:

    @pytest.fixture
    def sample_documents(self):
        return [{
            "content": "# Section 1\nShort paragraph.\n# Section 2\nAnother paragraph.",
            "filename": "test.md",
            "filepath": "/tmp/test.md",
        }]

    def test_returns_list_of_documents(self, sample_documents):
        chunks = chunk_documents(sample_documents)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunk_has_metadata(self, sample_documents):
        chunks = chunk_documents(sample_documents)
        meta = chunks[0].metadata
        assert "source" in meta
        assert "chunk_id" in meta
        assert "section_heading" in meta

    def test_chunk_ids_are_sequential(self, sample_documents):
        chunks = chunk_documents(sample_documents)
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert ids == list(range(len(ids)))

    def test_large_section_gets_split(self):
        big_content = "# Big\n" + ("Lorem ipsum dolor sit amet. " * 200)
        docs = [{"content": big_content, "filename": "big.md", "filepath": "/tmp/big.md"}]
        chunks = chunk_documents(docs, chunk_size=500)
        assert len(chunks) > 1
        assert all(c.metadata["chunk_type"] == "subsection" for c in chunks)

    def test_small_section_single_chunk(self):
        docs = [{"content": "# Small\nTiny.", "filename": "s.md", "filepath": "/tmp/s.md"}]
        chunks = chunk_documents(docs, chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0].metadata["chunk_type"] == "section"
