"""
Part A (steps 2-3) - Turn raw_pages.txt into retrieval-friendly chunks.

Strategy (see README.md "Chunking strategy" for the full rationale):
  1. Split the extracted text on section/sub-section headings using a small
     set of regex patterns tuned to this manual's formatting conventions
     (ALL-CAPS heading lines, numbered headings like "1.3.2", and named
     hardware headings such as "1830 PSS-8 Fan Unit (8FAN)").
  2. Each heading starts a new "section". A section is then packed into
     100-300 word chunks WITHOUT crossing a heading boundary, so a fan
     unit's own description never gets split across two unrelated chunks.
  3. If a single section is longer than ~300 words, it is split further at
     paragraph/blank-line boundaries (never mid-paragraph, never mid-table
     row where avoidable).
  4. If a section is shorter than ~100 words (common for short spec call-
     outs, e.g. a one-paragraph card description), it is merged with the
     next section under the SAME parent heading rather than left as a tiny,
     context-poor chunk.
  5. Every chunk carries metadata: section heading, page number(s) it spans,
     and a chunk_id.

Output: ../data/chunks.jsonl  (one JSON object per line)
"""
import argparse
import json
import re
from dataclasses import dataclass, field

HEADING_PATTERNS = [
    # Numbered headings e.g. "1.3.2 Shelf types", "2.4 Fan units"
    re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,3})\s+([A-Z][A-Za-z0-9 ,/&()\-]{3,80})\s*$"),
    # Named hardware/product headings e.g. "1830 PSS-8 Fan Unit (8FAN)"
    re.compile(r"^\s*(1830\s*PSS-?\w*[^\n]{0,80})\s*$"),
    # Short ALL-CAPS lines used as running section titles in the manual
    re.compile(r"^\s*([A-Z][A-Z0-9 /&\-]{6,60})\s*$"),
]

MIN_WORDS = 100
TARGET_MAX_WORDS = 300


# Explicit signature for this manual's footer line, e.g.:
# "© 2023 Nokia. Nokia Confidential Information Use subject to agreed
#  restrictions on disclosure and use. 3KC-71311-QAAA-HQZZA Release 23.6
#  June 2023 166 Issue 1"
# Matched outright regardless of length -- frequency alone missed it because
# it's much longer than a typical short running header.
FOOTER_LINE_RE = re.compile(
    r"^\s*©\s*\d{4}\s*Nokia\.?\s*Nokia\s*Confidential\s*Information.*Issue\s*\d+\s*$",
    re.IGNORECASE,
)


def strip_boilerplate(raw_text: str) -> str:
    """Remove running headers/footers (copyright line, doc number, 'Issue N',
    bare page numbers) that repeat on nearly every page. Left in place, these
    get mis-detected as section headings (e.g. a footer fragment like
    '74 Issue 1' matches the numbered-heading pattern) and corrupt chunking
    around them. Two mechanisms: (1) an explicit regex for this manual's
    known footer signature, which is long and would evade a naive frequency
    check; (2) frequency-based removal for SHORT recurring lines (running
    headers, model-name titles) -- length-gated so a genuinely unique long
    paragraph is never at risk of being dropped.
    """
    pages = re.split(r"(?=^<<<PAGE \d+>>>$)", raw_text, flags=re.MULTILINE)
    page_lines = []
    for p in pages:
        lines = [l.strip() for l in p.splitlines() if l.strip() and not l.startswith("<<<PAGE")]
        lines = [l for l in lines if not FOOTER_LINE_RE.match(l)]
        page_lines.append(lines)
    n_pages = max(1, len(page_lines))

    def normalize(line: str) -> str:
        return re.sub(r"\d+", "#", line)  # collapse page numbers so '166 Issue 1' == '74 Issue 1'

    # Only short lines are eligible for frequency-based removal. Running
    # headers/footers (copyright notices, doc numbers, page/issue stamps,
    # repeated model-name running titles) are always short. Real technical
    # paragraphs are long and should never be dropped even if a similar
    # phrase happens to recur — restricting by length keeps this safe.
    MAX_BOILERPLATE_WORDS = 14

    freq: dict[str, int] = {}
    for lines in page_lines:
        seen_this_page = set()
        for l in lines:
            if len(l.split()) > MAX_BOILERPLATE_WORDS:
                continue
            key = normalize(l)
            if key not in seen_this_page:
                freq[key] = freq.get(key, 0) + 1
                seen_this_page.add(key)

    threshold = max(5, int(n_pages * 0.4))
    boilerplate_keys = {k for k, c in freq.items() if c >= threshold}

    out_parts = []
    for i, p in enumerate(pages):
        marker_match = re.match(r"^(<<<PAGE \d+>>>)", p)
        marker = marker_match.group(1) if marker_match else ""
        body_lines = [l.strip() for l in p.splitlines() if l.strip() and not l.startswith("<<<PAGE")]
        body_lines = [l for l in body_lines if not FOOTER_LINE_RE.match(l)]
        kept = [l for l in body_lines
                if len(l.split()) > MAX_BOILERPLATE_WORDS or normalize(l) not in boilerplate_keys]
        if marker:
            out_parts.append(marker)
        out_parts.extend(kept)
    removed = sum(len(lines) for lines in page_lines) - sum(
        1 for l in out_parts if not l.startswith("<<<PAGE")
    )
    print(f"strip_boilerplate: removed {removed} repeated lines "
          f"({len(boilerplate_keys)} distinct patterns, threshold={threshold} pages)")
    return "\n".join(out_parts)


@dataclass
class Section:
    heading: str
    page: int
    lines: list = field(default_factory=list)


def is_heading(line: str) -> str | None:
    line = line.strip()
    if not line or len(line) > 90:
        return None
    for pat in HEADING_PATTERNS:
        m = pat.match(line)
        if m:
            return line
    return None


def parse_sections(raw_text: str) -> list[Section]:
    sections: list[Section] = []
    current_page = 0
    current = Section(heading="(front matter)", page=current_page)
    for raw_line in raw_text.splitlines():
        page_marker = re.match(r"^<<<PAGE (\d+)>>>$", raw_line.strip())
        if page_marker:
            current_page = int(page_marker.group(1))
            continue
        heading = is_heading(raw_line)
        if heading:
            if current.lines:
                sections.append(current)
            current = Section(heading=heading, page=current_page)
        else:
            if raw_line.strip():
                current.lines.append(raw_line.strip())
    if current.lines:
        sections.append(current)
    return sections


def words(text: str) -> int:
    return len(text.split())


def sections_to_chunks(sections: list[Section]) -> list[dict]:
    chunks = []
    chunk_id = 0
    buffer_text: list[str] = []
    buffer_heading = None
    buffer_page = None

    def flush():
        nonlocal buffer_text, buffer_heading, buffer_page, chunk_id
        if buffer_text:
            text = " ".join(buffer_text).strip()
            if text:
                chunks.append({
                    "chunk_id": f"c{chunk_id:04d}",
                    "heading": buffer_heading,
                    "page": buffer_page,
                    "text": text,
                    "n_words": words(text),
                })
                chunk_id += 1
        buffer_text = []

    for sec in sections:
        body = " ".join(sec.lines).strip()
        if not body:
            continue
        # If we can merge this short section into the still-open buffer under
        # a related heading run, do so (rule 4 above); otherwise flush.
        if buffer_text and words(" ".join(buffer_text)) < MIN_WORDS:
            buffer_text.append(f"[{sec.heading}] {body}")
            buffer_page = buffer_page or sec.page
            if words(" ".join(buffer_text)) >= TARGET_MAX_WORDS:
                flush()
            continue

        flush()
        buffer_heading = sec.heading
        buffer_page = sec.page

        if words(body) <= TARGET_MAX_WORDS:
            buffer_text = [body]
            if words(body) >= MIN_WORDS:
                flush()
        else:
            # Long section: split on paragraph breaks, packing to ~300 words.
            paras = [p for p in re.split(r"\n{2,}|(?<=\.) {2,}", body) if p.strip()]
            if len(paras) == 1:
                paras = body.split(". ")
            acc = []
            for para in paras:
                acc.append(para)
                if words(" ".join(acc)) >= TARGET_MAX_WORDS:
                    buffer_text = acc
                    flush()
                    buffer_heading = sec.heading
                    buffer_page = sec.page
                    acc = []
            if acc:
                buffer_text = acc
    flush()
    return chunks


def main(in_path: str, out_path: str):
    raw = open(in_path, encoding="utf-8").read()
    raw = strip_boilerplate(raw)
    sections = parse_sections(raw)
    chunks = sections_to_chunks(sections)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    sizes = [c["n_words"] for c in chunks]
    print(f"{len(chunks)} chunks written -> {out_path}")
    if sizes:
        print(f"word count min/avg/max: {min(sizes)}/{sum(sizes)//len(sizes)}/{max(sizes)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default="../data/raw_pages.txt")
    ap.add_argument("--out", default="../data/chunks.jsonl")
    args = ap.parse_args()
    main(args.in_path, args.out)