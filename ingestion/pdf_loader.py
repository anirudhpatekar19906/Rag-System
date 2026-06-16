"""
pdf_loader.py
-------------
Extracts text from a PDF file page by page using PyMuPDF (fitz).

Returns a list of dicts:
    [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]
"""

import fitz  # PyMuPDF


def load_pdf(pdf_path: str) -> list[dict]:
    """
    Open a PDF and extract text from every page.

    Args:
        pdf_path: Path to the .pdf file.

    Returns:
        List of {"page": int, "text": str, "source": str} dicts.
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:  # skip blank pages
            pages.append({
                "page": page_num,
                "text": text,
                "source": f"{pdf_path}:page{page_num}",
            })

    doc.close()
    print(f"[pdf_loader] Loaded {len(pages)} pages from '{pdf_path}'")
    return pages


def load_pdf_as_text(pdf_path: str) -> str:
    """
    Convenience wrapper — returns all pages joined as a single string.
    Useful when you just want the raw text blob.
    """
    pages = load_pdf(pdf_path)
    return "\n\n".join(p["text"] for p in pages)


# ── quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_loader.py <path_to_pdf>")
        sys.exit(1)

    path = sys.argv[1]
    result = load_pdf(path)
    for entry in result[:3]:          # preview first 3 pages
        print(f"\n--- Page {entry['page']} ---")
        print(entry["text"][:500])    # first 500 chars