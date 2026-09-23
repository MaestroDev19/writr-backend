from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Any, Literal

import re
import unicodedata

import tiktoken
from fastapi import Depends
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from core.config import get_settings

settings = get_settings()

Profile = Literal["atomic", "record", "structured", "prose"]

# (chunk text, profile-specific metadata extras)
Piece = tuple[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class Chunk:
    content: str
    index: int
    profile: Profile
    metadata: dict


class Chunker:
    def __init__(
        self,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        atomic_max: int | None = None,
        min_chunk: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size  # e.g. 2000
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap  # 0
        self.atomic_max = atomic_max or settings.atomic_max
        self.min_chunk = min_chunk or settings.min_chunk

        # Structure pass (markdown headings) — used by split_markdown
        self._md = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=False,
        )
        # Size fallback — recursive separators when a piece is still too big
        self._size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        )

    def tokenCount(self, text: str, safety: float = 0.75) -> int:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        return int(len(tokens) * safety)

    def normalize(self, text: str) -> str:
        # 1. unicode / invisible junk
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\u00ad", "")  # soft hyphen
        text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
        # 2. line endings + whitespace
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 3. PDF line-break hyphenation: "gov-\nernment" → "government"
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        # 4. curly quotes → straight (helps quote-balance lint)
        text = text.translate(
            str.maketrans(
                {
                    "\u2018": "'",
                    "\u2019": "'",
                    "\u201c": '"',
                    "\u201d": '"',
                }
            )
        )
        # 5. scene breaks → one sentinel for prose splitting
        text = re.sub(
            r"(?m)^\s*(\*\*\*|---|· · ·|# # #)\s*$",
            "\n\n***\n\n",
            text,
        )
        return text.strip()

    def has_markdown_headings(self, text: str) -> bool:
        return bool(re.search(r"(?m)^#{1,6}\s+\S", text))

    def looks_like_record(self, text: str) -> bool:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return False
        keyed = sum(
            1 for ln in lines if re.match(r"^[^:\n]{1,40}:\s+\S", ln)
        )
        return keyed / len(lines) >= 0.30

    def strip_repeated_lines(self, text: str) -> str:
        lines = text.splitlines()
        if not lines:
            return text
        first_line = lines[0]
        return text.replace(first_line, "")

    def determine_profile(self, text: str) -> Profile:
        text = self.normalize(text)
        tokenCount = self.tokenCount(text)
        if tokenCount < self.min_chunk:
            return "atomic"
        elif tokenCount < self.atomic_max or self.looks_like_record(text):
            return "record"
        elif self.has_markdown_headings(text):
            return "structured"
        else:
            return "prose"

    def split_markdown(self, text: str) -> list[Piece]:
        """Headers first; size-split oversized sections. Keeps heading_path."""
        docs = self._md.split_text(text)  # list of Documents with metadata
        parts: list[Piece] = []
        for doc in docs:
            body = doc.page_content
            # Build breadcrumb from MarkdownHeaderTextSplitter metadata
            heading_path = [
                doc.metadata[key]
                for key in ("h1", "h2", "h3")
                if key in doc.metadata and doc.metadata[key]
            ]
            extra: dict[str, Any] = {}
            if heading_path:
                extra["heading_path"] = heading_path

            if len(body) <= self.chunk_size:
                parts.append((body, extra))
            else:
                # Same heading_path on every sub-chunk from this section
                for piece in self._size_splitter.split_text(body):
                    parts.append((piece, dict(extra)))
        return parts

    def split_prose(self, text: str) -> list[Piece]:
        """Scene breaks when present; else whole text → size splitter fallback."""
        scenes = re.split(r"\n\n\*\*\*\n\n", text)
        parts: list[Piece] = []
        for scene_index, scene in enumerate(scenes):
            scene = scene.strip()
            if not scene:
                continue
            extra: dict[str, Any] = {"scene_index": scene_index}
            if len(scene) <= self.chunk_size:
                parts.append((scene, extra))
            else:
                # Fallback: no useful structure left — split by size
                for piece in self._size_splitter.split_text(scene):
                    parts.append((piece, dict(extra)))
        return parts

    def split_records(self, text: str) -> list[Piece]:
        """One block per blank-line-separated entity; attach entity name if found."""
        blocks = re.split(r"\n\s*\n", text)
        parts: list[Piece] = []
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            extra: dict[str, Any] = {}
            entity = self._entity_from_record(block)
            if entity:
                extra["entity"] = entity
            parts.append((block, extra))
        return parts

    def _entity_from_record(self, block: str) -> str | None:
        """Best-effort Name:/Title: (or first line) for record metadata."""
        for line in block.splitlines():
            stripped = line.strip()
            match = re.match(r"(?i)^(?:name|title)\s*:\s*(.+)$", stripped)
            if match:
                return match.group(1).strip()
        # Fallback: first non-empty line as a weak label
        for line in block.splitlines():
            if line.strip():
                return line.strip()[:80]
        return None

    def _merge_small_tails(self, pieces: list[Piece]) -> list[Piece]:
        """Merge orphan tails shorter than min_chunk into the previous piece."""
        if not pieces:
            return pieces
        merged: list[Piece] = [pieces[0]]
        for content, extra in pieces[1:]:
            prev_content, prev_extra = merged[-1]
            if len(content) < self.min_chunk:
                merged[-1] = (f"{prev_content}\n\n{content}", prev_extra)
            else:
                merged.append((content, extra))
        return merged

    def chunk(self, text: str, *, doc_name: str | None = None) -> list[Chunk]:
        """
        Public entry: normalize → profile once → split → merge tiny tails → Chunks.

        Profile is chosen once for the whole document (not per fragment).
        Size safety net lives inside split_* via _size_splitter.
        """
        text = self.normalize(text)
        if not text:
            return []

        profile = self.determine_profile(text)

        # Route by profile (flat dispatch — not recursive)
        if profile == "atomic":
            pieces: list[Piece] = [(text, {})]
        elif profile == "record":
            pieces = self.split_records(text)
        elif profile == "structured":
            pieces = self.split_markdown(text)
        else:
            pieces = self.split_prose(text)

        pieces = self._merge_small_tails(pieces)

        chunks: list[Chunk] = []
        for index, (content, extra) in enumerate(pieces):
            # Shared base metadata + profile-specific extras
            metadata: dict[str, Any] = {
                "profile": profile,
                "doc_name": doc_name,
                "char_count": len(content),
                **extra,
            }
            chunks.append(
                Chunk(
                    content=content,
                    index=index,
                    profile=profile,
                    metadata=metadata,
                )
            )
        return chunks


@lru_cache(maxsize=1)
def get_chunker() -> Chunker:
    """Cached factory for FastAPI dependency injection."""
    return Chunker()


ChunkerDep = Annotated[Chunker, Depends(get_chunker)]
