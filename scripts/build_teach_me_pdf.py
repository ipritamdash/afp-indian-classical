"""Build an AFP doc PDF from a markdown source. Fully automated, idempotent.

Two presets available out of the box:

  /tmp/venv_metal/bin/python scripts/build_teach_me_pdf.py            # deep guide
  /tmp/venv_metal/bin/python scripts/build_teach_me_pdf.py --doc overview   # overview

Or pass arbitrary paths:

  ... --src path/to/some.md --out path/to/out.pdf --title "..." --subtitle "..."

Outputs the PDF plus a sibling .meta.json with the build manifest.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sys
import time
from pathlib import Path

# markdown_pdf must be available in the running Python env
try:
    from markdown_pdf import MarkdownPdf, Section
except ImportError:
    sys.stderr.write(
        "ERROR: markdown_pdf not installed in this Python.\n"
        "Run with the metal venv:  /tmp/venv_metal/bin/python scripts/build_teach_me_pdf.py\n"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Doc presets
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

PRESETS = {
    "teach": {
        "src":      REPO_ROOT / "docs" / "teach_me.md",
        "out":      REPO_ROOT / "docs" / "AFP_Teaching_Guide.pdf",
        "title":    "Audio Fingerprinting for Indian Classical Music",
        "subtitle": "A teach-me-everything personal guide to your B.Tech-II project",
        "cover_blurb": (
            "<p><b>Scope.</b> Five audio-fingerprinting systems benchmarked on Saraga 1.5 "
            "(357 Indian classical refs, 6 528 evaluation cells). Plus a pre-registered "
            "training-recipe improvement to NAFP with Bonferroni-significant gains: pooled "
            "McNemar p = 3.18 × 10⁻⁶ on the hardest 1-second cell, ~68 % miss-rate reduction.</p>"
            "<p><b>How to read this doc.</b> Each topic has a green \"🟢 Must Know\" essentials "
            "block and a blue \"🔵 Depth\" details block. The last section (Part 27) is a one-page "
            "cheat sheet of every number, name, and formula.</p>"
        ),
    },
    "overview": {
        "src":      REPO_ROOT / "docs" / "overview.md",
        "out":      REPO_ROOT / "docs" / "AFP_Overview_Guide.pdf",
        "title":    "Audio Fingerprinting for Indian Classical Music",
        "subtitle": "A high-level overview — concepts, pipelines, and the questions they'll ask you",
        "cover_blurb": (
            "<p><b>This doc is the friendly overview.</b> "
            "Pipeline-focused. Concepts, not formulas. After every Part there's a "
            "<i>\"Questions they'll ask\"</i> box with the most likely viva / interview / advisor "
            "questions and ready answers.</p>"
            "<p><b>If you want the math + derivations</b> (NT-Xent loss, McNemar test, Wilson CIs, etc.), "
            "see the companion doc <code>AFP_Teaching_Guide.pdf</code>.</p>"
            "<p><b>The headline.</b> 5-system benchmark on Saraga 1.5. NAFP baseline at 98.3 % HR@1 "
            "(1 sec), recipe v3 at 99.5 % (~68 % fewer mistakes), statistically validated, "
            "published with a live calibrated demo.</p>"
        ),
    },
    "engineering": {
        "src":      REPO_ROOT / "docs" / "engineering_log.md",
        "out":      REPO_ROOT / "docs" / "AFP_Engineering_Log.pdf",
        "title":    "Engineering Log — What we tried before Recipe v3 worked",
        "subtitle": "Failures, audits, bugs, and the discipline that made the headline publishable",
        "cover_blurb": (
            "<p><b>This is the engineering slice of the project.</b> "
            "Not the headline numbers — the work that made the headline numbers <i>defensible</i>. "
            "Three major experiments that did not become the headline (Recipe v2 intermediate; "
            "Intervention 2 negative; hubness post-processing negative), the Phase 1 audit that "
            "caught four real latent bugs, three engineering gotchas that ate hours and are now "
            "permanent memory notes, and the methodological choices (pre-registration, multiple "
            "controls, cross-corpus tuning) that anchor every claim.</p>"
            "<p><b>Each item is sized in proportion to how much intellectual work it represented.</b> "
            "Major experiments get full anatomy (Hypothesis / Design / Result / Lesson). Small fixes "
            "get a paragraph. Every claim cites a file path in the repo. No hallucinations.</p>"
            "<p><b>Companion docs.</b> Deep math: <code>AFP_Teaching_Guide.pdf</code>. "
            "High-level overview: <code>AFP_Overview_Guide.pdf</code>.</p>"
        ),
    },
}

AUTHOR_LINE = "Aryan Banwala · DTU CSE · B.Tech-II"

# ---------------------------------------------------------------------------
# CSS — one shared stylesheet applied to every section
# ---------------------------------------------------------------------------

CSS = """
@page {
    size: A4;
    margin: 18mm 16mm 18mm 16mm;
}

* { box-sizing: border-box; }

body {
    font-family: "Helvetica Neue", "Arial", sans-serif;
    font-size: 10.5pt;
    line-height: 1.55;
    color: #1a1a1a;
}

/* Headings */
h1 {
    font-size: 22pt;
    color: #102a43;
    border-bottom: 2px solid #102a43;
    padding-bottom: 6pt;
    margin-top: 6pt;
    margin-bottom: 12pt;
    page-break-before: auto;
}
h2 {
    font-size: 16pt;
    color: #1d4e89;
    margin-top: 14pt;
    margin-bottom: 8pt;
    border-bottom: 1px solid #d0d7de;
    padding-bottom: 3pt;
}
h3 {
    font-size: 13pt;
    color: #1d4e89;
    margin-top: 12pt;
    margin-bottom: 6pt;
}
h4 {
    font-size: 11pt;
    color: #243b53;
    margin-top: 10pt;
    margin-bottom: 5pt;
    font-weight: 700;
}

p { margin: 0 0 8pt 0; text-align: left; }

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 8pt 0 12pt 0;
    font-size: 9.5pt;
}
th, td {
    border: 1px solid #c8d1da;
    padding: 4pt 6pt;
    text-align: left;
    vertical-align: top;
}
th {
    background: #e8eef6;
    color: #102a43;
    font-weight: 700;
}
tr:nth-child(even) td { background: #f7f9fc; }

/* Inline + block code */
code {
    font-family: "Menlo", "Monaco", "Courier New", monospace;
    font-size: 9.5pt;
    background: #f4f5f7;
    padding: 1pt 3pt;
    border-radius: 3px;
    color: #243b53;
}
pre {
    font-family: "Menlo", "Monaco", "Courier New", monospace;
    font-size: 9pt;
    background: #f4f5f7;
    border: 1px solid #d6dde5;
    border-left: 3px solid #1d4e89;
    border-radius: 4px;
    padding: 8pt 10pt;
    margin: 8pt 0;
    white-space: pre-wrap;
    word-wrap: break-word;
    color: #243b53;
}
pre code { background: transparent; padding: 0; }

/* Blockquotes — used as "Questions they'll ask" callouts in the overview doc */
blockquote {
    border-left: 4px solid #d97706;
    background: #fef6e7;
    padding: 8pt 12pt;
    margin: 10pt 0;
    color: #1a3358;
    border-radius: 0 4px 4px 0;
}
blockquote p { margin: 0 0 6pt 0; }
blockquote p:last-child { margin-bottom: 0; }
blockquote strong { color: #92400e; }

/* Lists */
ul, ol { margin: 4pt 0 8pt 18pt; }
li { margin-bottom: 3pt; }

/* Links */
a { color: #1d4e89; text-decoration: none; }
a:hover { text-decoration: underline; }

/* Horizontal rules */
hr {
    border: none;
    border-top: 1px solid #c8d1da;
    margin: 12pt 0;
}

/* Cover styling */
.cover-title {
    font-size: 32pt;
    font-weight: 700;
    color: #102a43;
    margin-top: 80pt;
    margin-bottom: 10pt;
    line-height: 1.2;
}
.cover-subtitle {
    font-size: 14pt;
    color: #1d4e89;
    margin-bottom: 40pt;
    font-style: italic;
}
.cover-author {
    font-size: 11pt;
    color: #486581;
    margin-bottom: 4pt;
}
.cover-meta {
    margin-top: 60pt;
    font-size: 10pt;
    color: #627d98;
}
.cover-meta b { color: #243b53; }

/* Must-Know / Depth headers (emoji-driven, styled via h2/h3 already) */
"""

# ---------------------------------------------------------------------------
# Splitting strategy
# ---------------------------------------------------------------------------

def count_top_level_sections(md_text: str) -> int:
    """Count top-level "# " headings, skipping fenced code blocks."""
    h1_re = re.compile(r"^# (?!#)")
    in_fence = False
    n = 0
    for line in md_text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and h1_re.match(line):
            n += 1
    return n


def build_cover_html(title: str, subtitle: str, blurb_html: str) -> str:
    today = datetime.date.today().isoformat()
    return f"""
<div class="cover-title">{title}</div>
<div class="cover-subtitle">{subtitle}</div>
<div class="cover-author"><b>Author:</b> {AUTHOR_LINE}</div>
<div class="cover-author"><b>Generated:</b> {today}</div>

<div class="cover-meta">
{blurb_html}

<p><b>Reproducibility.</b> Every claim is grounded in a file in the repo
(<code>data/results/...</code>) or a peer-reviewed paper. Pre-registration protocols
are committed in git. No hallucinations.</p>

<p><b>Artifacts.</b><br>
&nbsp;&nbsp;Live demo · <code>huggingface.co/spaces/Tachyeon/afp-indian-classical-demo</code><br>
&nbsp;&nbsp;Dataset · <code>huggingface.co/datasets/Tachyeon/audio-fingerprint-indian-bench</code><br>
&nbsp;&nbsp;Code · <code>github.com/ipritamdash/afp-indian-classical</code></p>
</div>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--doc", choices=list(PRESETS.keys()), default="teach",
                    help="Built-in preset: 'teach' (deep guide) or 'overview' (high-level)")
    ap.add_argument("--src", type=Path, default=None,
                    help="Override: source markdown file")
    ap.add_argument("--out", type=Path, default=None,
                    help="Override: output PDF path")
    ap.add_argument("--title", type=str, default=None, help="Override: cover title")
    ap.add_argument("--subtitle", type=str, default=None, help="Override: cover subtitle")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    preset = PRESETS[args.doc]
    src_md = args.src or preset["src"]
    out_pdf = args.out or preset["out"]
    title = args.title or preset["title"]
    subtitle = args.subtitle or preset["subtitle"]
    blurb = preset["cover_blurb"]

    out_meta = out_pdf.with_suffix(".meta.json")
    t0 = time.time()

    # --- 1. Read source markdown ---
    if not src_md.exists():
        sys.stderr.write(f"ERROR: source markdown not found at {src_md}\n")
        return 1
    md_text = src_md.read_text(encoding="utf-8")
    md_sha = hashlib.sha256(md_text.encode("utf-8")).hexdigest()[:12]
    print(f"[1/4] Source: {src_md.relative_to(REPO_ROOT)}  "
          f"({len(md_text):,} chars, sha256={md_sha})")

    # --- 2. Count top-level sections (for the manifest) ---
    n_top_level = count_top_level_sections(md_text)
    print(f"[2/4] Detected {n_top_level} top-level sections in source")

    # --- 3. Build the PDF ---
    # Render the cover as one Section and the whole body as a single section
    # so that internal anchor links (#part-N) resolve correctly. markdown_pdf
    # auto-generates a clickable TOC from H1/H2 headings.
    pdf = MarkdownPdf(toc_level=2, optimize=True)

    pdf.meta["title"] = title
    pdf.meta["author"] = "Aryan Banwala"
    pdf.meta["subject"] = "B.Tech-II teaching document"
    pdf.meta["keywords"] = "audio fingerprinting, NAFP, NMFP, Saraga, Indian classical music"

    # Cover page (no TOC entry)
    cover_md = build_cover_html(title, subtitle, blurb)
    pdf.add_section(Section(cover_md, toc=False, paper_size="A4"), user_css=CSS)
    print(f"[3/4] Cover added; rendering body as a single section")

    # Body
    pdf.add_section(Section(md_text, paper_size="A4"), user_css=CSS)

    # --- 4. Save ---
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.save(str(out_pdf))
    out_size_mb = out_pdf.stat().st_size / (1024 * 1024)

    # Verify page count via pymupdf
    try:
        import pymupdf
        with pymupdf.open(str(out_pdf)) as doc:
            n_pages = doc.page_count
    except Exception as e:
        n_pages = -1
        print(f"      [warn] page count probe failed: {e}")

    meta = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "preset": args.doc,
        "source_md": str(src_md.relative_to(REPO_ROOT)),
        "source_sha256_12": md_sha,
        "source_chars": len(md_text),
        "n_sections": n_top_level,
        "output_pdf": str(out_pdf.relative_to(REPO_ROOT)),
        "output_size_mb": round(out_size_mb, 2),
        "n_pages": n_pages,
        "build_time_sec": round(time.time() - t0, 2),
        "tool": "markdown_pdf",
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[4/4] Wrote {out_pdf.relative_to(REPO_ROOT)}  "
          f"({out_size_mb:.2f} MB, {n_pages} pages, {time.time()-t0:.1f}s)")
    print(f"      Metadata at {out_meta.relative_to(REPO_ROOT)}")
    print()
    print(f"  Open with:  open '{out_pdf}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
