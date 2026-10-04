"""Download the corpus PDFs listed in data/corpus/papers.yaml from arXiv.

Standard library only, so it runs with any Python 3.9+ (no venv needed):

    python scripts/download_corpus.py

Already-downloaded files are skipped. Waits between requests to respect arXiv's rate limits.
"""

import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "corpus" / "papers.yaml"
OUT = ROOT / "data" / "corpus" / "pdf"
DELAY_S = 3


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Tiny parser for the flat `- id: ... / key: value` list, so PyYAML is not required."""
    papers: list[dict[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split(" #", 1)[0].rstrip() if not raw.lstrip().startswith("#") else ""
        m = re.match(r"^\s*(-\s+)?(\w[\w-]*):\s*(.*)$", line)
        if not m or m.group(2) == "papers":
            continue
        is_new, key, value = m.group(1), m.group(2), m.group(3).strip().strip('"')
        if is_new:
            papers.append({})
        if papers:
            papers[-1][key] = value
    return papers


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    papers = read_manifest(MANIFEST)
    manual, failed = [], []
    for paper in papers:
        pid = paper["id"]
        dest = OUT / f"{pid}.pdf"
        if dest.exists() and dest.stat().st_size > 10_000:
            print(f"  have  {pid}")
            continue
        if "arxiv" not in paper:
            manual.append(paper)
            continue
        url = f"https://arxiv.org/pdf/{paper['arxiv']}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "papertrail-corpus/0.1"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if not data.startswith(b"%PDF"):
                raise ValueError("response is not a PDF")
            dest.write_bytes(data)
            print(f"  got   {pid}  ({len(data) // 1024} KB)")
        except Exception as exc:
            failed.append(pid)
            print(f"  FAIL  {pid}: {exc}")
        time.sleep(DELAY_S)

    have = len(list(OUT.glob("*.pdf")))
    print(f"\n{have}/{len(papers)} PDFs in {OUT}")
    for paper in manual:
        print(f"  manual: save '{paper['title']}' as {OUT / (paper['id'] + '.pdf')}")
    if failed:
        print(f"  failed (re-run to retry): {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
