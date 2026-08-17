#!/usr/bin/env python3
"""Render RESULTS.md into a self-contained RESULTS.html (figure embedded as
base64), so the doc displays identically in any browser or mail client."""

import base64
import re
from pathlib import Path

import markdown

import sys

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
md = (ROOT / "RESULTS.md").read_text(encoding="utf-8")

def embed(match):
    path = ROOT / match.group(2)
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f'<img alt="{match.group(1)}" src="data:image/png;base64,{b64}" style="max-width:100%">'

md = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", embed, md)
body = markdown.markdown(md, extensions=["tables", "fenced_code"])
css = """
body{font:15px/1.55 -apple-system,'Segoe UI',sans-serif;color:#1f2937;max-width:780px;
margin:2rem auto;padding:0 1rem}h1{font-size:1.5rem}h2{font-size:1.15rem;margin-top:1.6em}
pre{background:#f3f4f6;padding:.8em;border-radius:6px;overflow-x:auto;font-size:12.5px}
code{background:#f3f4f6;padding:.1em .3em;border-radius:4px;font-size:.9em}
table{border-collapse:collapse;margin:.8em 0}td,th{border:1px solid #d1d5db;padding:.35em .6em;text-align:left}
"""
html = f"<!doctype html><meta charset='utf-8'><title>Probing results</title><style>{css}</style>{body}"
(ROOT / "RESULTS.html").write_text(html, encoding="utf-8")
print(f"-> {ROOT / 'RESULTS.html'} ({len(html)//1024} KB)")
