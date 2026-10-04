"""Download the embedding and reranker models into models/ (git-ignored).

Standard library only, so it runs with any Python 3.9+:

    python scripts/download_models.py

Fetches the exact files sentence-transformers needs from Hugging Face, pinned to a revision so
every run (and every ablation number) uses the same weights. Already-downloaded files are skipped.
"""

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models"

MODELS = {
    # Embedding model: 33M params, 384-dim vectors (matches VECTOR(384) in scripts/init.sql).
    "BAAI/bge-small-en-v1.5": [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "modules.json",
        "sentence_bert_config.json",
        "config_sentence_transformers.json",
        "1_Pooling/config.json",
    ],
    # Cross-encoder reranker: 22M params, trained on MS MARCO passage ranking.
    "cross-encoder/ms-marco-MiniLM-L-6-v2": [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
    ],
}
REVISION = "main"


def main() -> int:
    failed = []
    for repo, files in MODELS.items():
        target = OUT / repo.replace("/", "__")
        for name in files:
            dest = target / name
            if dest.exists() and dest.stat().st_size > 0:
                print(f"  have  {repo}/{name}")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/{repo}/resolve/{REVISION}/{name}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "papertrail/0.1"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = resp.read()
                dest.write_bytes(data)
                print(f"  got   {repo}/{name}  ({len(data) // 1024} KB)")
            except Exception as exc:
                failed.append(f"{repo}/{name}")
                print(f"  FAIL  {repo}/{name}: {exc}")
    print(f"\nModels in {OUT}")
    if failed:
        print("  failed (re-run to retry): " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
