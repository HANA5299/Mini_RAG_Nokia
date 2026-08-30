"""
Part A (step 1) - Extract Chapters 1-2 (pages 47-166) from the Nokia 1830 PSS
Product Information & Planning Guide into a single plain-text file, with a
page-boundary marker before every page so downstream chunking can attach a
correct page number to each chunk.

Usage:
    python 01_extract.py --pdf /path/to/1830_Technical_Description.pdf \
                          --start 47 --end 166 \
                          --out ../data/raw_pages.txt
"""
import argparse
from pypdf import PdfReader


def extract(pdf_path: str, start: int, end: int, out_path: str) -> None:
    reader = PdfReader(pdf_path)
    n_pages = len(reader.pages)
    if end > n_pages:
        raise ValueError(f"--end {end} exceeds document length ({n_pages} pages)")

    with open(out_path, "w", encoding="utf-8") as f:
        # PDF page numbers in the guide vs. pypdf's 0-indexed array can be
        # offset by front matter (cover, TOC, revision history). We extract
        # by *physical* page index here; see README for how we reconciled
        # the printed page numbers in the header/footer with pypdf's index.
        for i in range(start - 1, end):
            page = reader.pages[i]
            text = page.extract_text() or ""
            # Marker consumed by 02_chunk.py to track page numbers.
            f.write(f"\n<<<PAGE {i + 1}>>>\n")
            f.write(text)
            f.write("\n")

    print(f"Extracted physical pages {start}-{end} ({end - start + 1} pages) -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to 1830_Technical_Description.pdf")
    ap.add_argument("--start", type=int, default=47)
    ap.add_argument("--end", type=int, default=166)
    ap.add_argument("--out", default="../data/raw_pages.txt")
    args = ap.parse_args()
    extract(args.pdf, args.start, args.end, args.out)
