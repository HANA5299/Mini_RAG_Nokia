# Nokia 1830 PSS Site-Engineering RAG Assistant

## What this is
A small Retrieval-Augmented Generation pipeline that answers site-engineering
questions strictly from Chapters 1-2 (physical pages 47-166) of the 1830 PSS
Product Information and Planning Guide, Release 23.6. It cites the section
heading and page it drew every claim from, and replies with the exact string
"Not found in the provided document." — never a guess — when the retrieved
context doesn't actually answer the question.

## Pipeline
```
01_extract.py         PDF pages 47-166 -> data/raw_pages.txt (page-tagged)         Part A (step 1)
02_chunk.py            raw_pages.txt   -> data/chunks.jsonl  (section-aware)        Part A (steps 2-3)
03_embed_index.py      chunks.jsonl    -> index/ (embeddings + metadata)            Part B
04_retrieve_query.py   question -> top-k chunks -> grounded prompt -> answer (CLI)  Part C
05_evaluate.py         runs the 8 fixed questions -> eval/results.md, results.json  Part D
app.py                 Streamlit UI wrapping the same retrieval + prompt logic      (bonus, optional)
```

All five numbered scripts are run from inside `scripts/`, in that order —
each one's output feeds the next.

## Chunking strategy (Part A)
The manual is organized around named headings — numbered section headings
("1.3.2 Shelf types") and standalone hardware headings ("1830 PSS-8 Fan Unit
(8FAN)"). Splitting purely by character/word count risks cutting a shelf's
slot count from the paragraph that names the shelf, so `02_chunk.py` instead:

1. **Strips running headers/footers first.** The manual repeats a copyright
   footer ("© 2023 Nokia... Release 23.6 June 2023 <page> Issue 1") and
   running page-header titles (e.g. "PSI-4L/PSI-8L") on nearly every page.
   Left in place, fragments of these were being mis-detected as real section
   headings by the numbered-heading pattern (e.g. "166 Issue 1" parses as
   "section 166, titled Issue 1"), which fragmented and corrupted chunking
   around them. `strip_boilerplate()` removes these via (a) an explicit
   regex for this manual's exact footer signature, regardless of length, and
   (b) a frequency check for short lines that recur across a large fraction
   of pages — deliberately length-gated so a genuinely unique long paragraph
   can never be at risk of being dropped.
2. Detects heading lines with three regex patterns tuned to this manual's
   formatting (numbered headings, named hardware headings, short ALL-CAPS
   running titles).
3. Treats each detected heading as a section boundary chunks never cross.
4. Packs each section's text into chunks of roughly 100-300 words; a
   section longer than ~300 words is split at paragraph breaks, never
   mid-sentence.
5. A section shorter than 100 words (a short spec call-out, e.g. "Rack
   mounting options") is merged into the buffer under the *previous*
   section's heading rather than left as a near-empty, low-context chunk.
   The merged sub-heading is preserved inline as a `[Sub-heading]` tag
   inside the chunk text, but the chunk's stored `heading` metadata field
   keeps the original heading. See Known Limitations — this is the source
   of most citation imprecision found during evaluation.
6. Every chunk stores `heading`, `page`, and `chunk_id` metadata.

**Actual result on the real document:** 112 chunks, word count
min/avg/max = 101/196/636. The `strip_boilerplate()` fix alone removed
1,095 repeated lines across 11 distinct recurring patterns on the first
real run — before that fix, the same pipeline produced 142 chunks with a
23-word minimum and a 539-word maximum, both symptoms of footer/header
contamination fragmenting and re-merging real content incorrectly.

## Embedding & indexing (Part B)
`sentence-transformers/all-MiniLM-L6-v2` (384-dim), L2-normalized so a plain
dot product equals cosine similarity. The manual top-k search in
`04_retrieve_query.py` and `app.py` is a single `embeddings @ query_vector`
+ `argsort` — no FAISS/Chroma dependency needed for 112 chunks. Embeddings
are cached under `index/` (`embeddings.npy`, `metadata.jsonl`,
`manifest.json`) keyed to a hash of `chunks.jsonl`, so re-running query
scripts doesn't re-embed everything; `03_embed_index.py` only needs to be
re-run when `chunks.jsonl` actually changes.

## Chosen k
**k = 4** (default in `04_retrieve_query.py` and `app.py`'s sidebar
slider). Most answers live in a single chunk, but several questions combine
two related facts (e.g. slot count *and* RU footprint) that can land in
adjacent-but-separate chunks after chunking. In testing, k=4 correctly
answered 7 of 8 fixed evaluation questions; the one miss (Q4) needed k=8-10
to surface the right chunk at all, since its retrieved chunk shared almost
no vocabulary with the question (see Evaluation below) — this is
documented as a retrieval-recall limitation rather than solved by simply
raising the default k, since a higher k also dilutes context and makes the
Q8 refusal less reliable in informal testing.

## System prompt (Part C)
The exact system prompt used (from `SYSTEM_PROMPT` in
`scripts/04_retrieve_query.py`, mirrored in `app.py`):

```
You are a Nokia 1830 PSS site-engineering assistant. You answer ONLY using
the CONTEXT chunks provided below, which are excerpts from the official
1830 PSS Product Information and Planning Guide.

Rules (follow all of them):
1. Base your answer strictly on the CONTEXT. Do not use outside knowledge
of Nokia products, and do not guess or estimate a plausible-sounding
number.
2. Every factual claim in your answer must be followed by a citation in
the form [Section: <heading>, p.<page>], using the heading/page metadata
attached to the chunk you drew it from.
3. If the CONTEXT does not contain the answer, reply EXACTLY: "Not found
in the provided document." — do not soften this into a guess, and do not
apologize at length. It is always better to say this than to invent a
number.
4. Keep answers concise: 1-4 sentences plus citations, unless the question
needs a list.
```

Generation is via Groq (`openai/gpt-oss-120b`, temperature 0) — swap this
in `call_groq()` for any other provider without touching retrieval or
grounding logic, which is entirely local code per the assignment's
requirement.

## Evaluation (Part D)
Run `python 05_evaluate.py` from inside `scripts/`; it produces
`eval/results.md` and `eval/results.json` with the retrieved chunk(s) and
generated answer for all 8 fixed questions.

**Every answer below was manually verified against the actual retrieved
chunk text in `data/chunks.jsonl`** (not just skimmed for plausibility) —
this surfaced real findings, not just a clean-looking table:

| # | Question (short) | Correct? | Notes |
|---|---|---|---|
| 1 | PSS-8 slots/RU | Yes | Matches manual verbatim (8-slot SWDM platform, 3-RU). An earlier pre-fix run of this same question returned "14 slots" from corrupted chunking — confirms the boilerplate-stripping fix was necessary, not cosmetic. |
| 2 | PSS-32 RU/slots | Yes | 14-RU, 32 slots both verbatim in source. Citation heading is mislabeled ("1830 PSS-16II shelf" instead of PSS-32) — the chunker's heading regex didn't recognize "2.5 1830 PSS-32 shelf" as a boundary, so the chunk inherited the previous section's heading label. Correct fact, wrong citation label. |
| 3 | Two load-lines | Yes | "SWDM software and OCS software" — exact match, clean citation. |
| 4 | PSS-32 fan units | Partial / No | At k=4, retrieval missed the source entirely and correctly returned "Not found" — but the content DOES exist (2.18.4 Front view, p.152: "PSS-32 Fan Units (FAN and FAN32H)"), so that refusal was a retrieval miss, not a correct grounded refusal. At k=8-10, retrieval found an adjacent chunk (2.18.1 Introduction) and generated an answer naming "FAN32H" — but that exact string does not appear anywhere in the retrieved chunk text. This is a genuine hallucination: the model added a plausible-sounding designation despite the system prompt's explicit "no outside knowledge, do not guess" rule. Confirmed by direct inspection of the chunk text. |
| 5 | PSS-16II fan unit | Yes | "PSS-16II FAN card" confirmed, merged under a `[2.17.1 Overview]` sub-tag inside a chunk headed "2.16.5 Visual indications" — same citation-imprecision pattern as Q2. |
| 6 | PSS-8 power filter cards | Yes | "8DC30, 8DC30T, 8DC30T2, 8AC7" — exact match to source text, clean citation. |
| 7 | PSS-8 rack aperture | Yes | 450.85mm/17.75in required, 444.5mm/17.5in explicitly unsupported — confirmed genuinely stated for PSS-8 (not borrowed from PSS-16/PSS-32, which independently state the identical value). Citation heading ("2.2.1 Introduction") is imprecise — the actual fact is merged in from a `[2.2.2 Rack mounting options]` sub-section. |
| 8 | Max optical reach (trick Q) | Yes | Correctly replied "Not found in the provided document." — this value is not in the extracted page range, and the pipeline did not guess a km figure. |

**Score: 7/8 correct** (one genuine hallucination on Q4). Both the pass
(Q8) and the failure (Q4) are documented with the exact chunk text that
proves each verdict, not just the model's own output.

## Known limitations
1. **Citation heading imprecision from section merging** (Q2, Q5, Q7). When
   a short section is folded into the preceding chunk's buffer (rule 5
   above), the chunk's stored `heading` metadata reflects the *first*
   section in the merged buffer, not the specific sub-heading a given fact
   actually came from — even though that sub-heading is preserved inline
   as a `[bracketed]` tag within the chunk text. Citations are
   chunk-accurate but not always sub-heading-precise. Affected 3 of 8 eval
   questions with correct facts but imprecise citations.
2. **Heading-regex misses cause heading mislabeling** (Q2). The numbered-
   heading pattern didn't recognize "2.5 1830 PSS-32 shelf" (no space
   between "2.5" and the shelf name in the source formatting) as a section
   boundary, so that whole section's content inherited the prior "1830
   PSS-16II shelf" heading label.
3. **Grounding is not absolute** (Q4). The system prompt's "context-only,
   no outside knowledge" rule reduces but does not eliminate hallucination
   risk — the model added a plausible but unretrieved model designation
   ("FAN32H") in one case despite explicit instructions not to. Stricter
   output constraints (e.g. requiring verbatim quotes for part/model
   numbers) would likely be needed to close this gap fully.
4. **Retrieval recall on low-vocabulary-overlap questions** (Q4). A
   question phrased generically ("which fan units are supported") can fail
   to surface a chunk whose heading/content shares almost no wording with
   the question (e.g. "2.18.4 Front view"), even though the answer is
   right there. Raising k to 8-10 sometimes surfaces it but isn't a
   guaranteed fix and increases the risk of diluting context for other
   questions.
5. **Diagram/figure legends extract as flattened, out-of-order text.**
   `pypdf` follows the PDF's internal content order rather than visual
   layout, so figure legends (e.g. Figure 2-24's network-connection
   labels: "EC_A", "10/100BASE-T", "CIT", etc.) come out as a jumbled
   run-on with no natural paragraph breaks. These chunks retrieve rarely
   since their embeddings don't closely match natural-language questions,
   but are a real extraction gap worth noting.
6. **Page numbers are pypdf's physical page index** passed to
   `--start`/`--end` (47-166), not necessarily the manual's own printed
   page number if front matter offsets them — not reconciled against the
   PDF's printed numbering in this build.
7. **No re-ranking or hybrid search by default** (see Part E for an
   optional add-on, not attempted in this submission) — retrieval is pure
   embedding similarity.
8. **Generation quality depends on the LLM in `call_groq()`** — currently
   `openai/gpt-oss-120b` via Groq's free tier (the original target,
   `llama-3.3-70b-versatile`, was deprecated by Groq mid-project). Swap by
   changing `GROQ_MODEL` and, if needed, the client call in that one
   function.

## Setup
```
pip install -r requirements.txt
```

Set your Groq API key (get a free one at console.groq.com):
```
$env:GROQ_API_KEY = "your-key-here"      # PowerShell, current session only
```

Run the full pipeline from inside `scripts/`:
```
cd scripts
python 01_extract.py --pdf ../data/1830_Technical_Description.pdf --start 47 --end 166 --out ../data/raw_pages.txt
python 02_chunk.py --in_path ../data/raw_pages.txt --out ../data/chunks.jsonl
python 03_embed_index.py --chunks ../data/chunks.jsonl --index_dir ../index
python 04_retrieve_query.py --q "How many slots does the 1830 PSS-8 shelf provide?"
python 05_evaluate.py
```

Optional Streamlit UI (same retrieval/prompt logic, interactive front-end):
```
pip install streamlit
streamlit run app.py
```
`app.py` reads its Groq key from a hardcoded `GROQ_API_KEY` constant near
the top of the file rather than an environment variable or UI prompt —
replace the placeholder with a real key before running, and do not commit
a real key if this repo is pushed anywhere public.

## Part E (stretch challenge)
Not attempted in this submission.
