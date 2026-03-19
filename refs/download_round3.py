#!/usr/bin/env python3
"""Round 3: fix remaining wrong/missing papers with known arXiv IDs."""
import os, time, requests

BASE = os.path.dirname(os.path.abspath(__file__))

# Manually verified arXiv IDs
PAPERS = [
    ("exposingguardrails_usenix25", "attack", "2412.00307"),
    ("trojanip_cvpr25", "attack", "2501.19388"),
    ("jailbreakdiffbench_iccv25", "benchmark", "2503.18573"),
    # Fix small/possibly wrong files
    ("art_neurips24", "attack", "2405.01860"),
    ("unlearndiffatk_eccv24", "attack", "2309.11907"),
    ("unsafebench_ccs25", "benchmark", "2405.11152"),
    ("t2isafety_cvpr25", "benchmark", "2501.12612"),
]


def download(arxiv_id, dest):
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    r = requests.get(url, timeout=30)
    if r.status_code == 200 and len(r.content) > 10000:
        with open(dest, "wb") as f:
            f.write(r.content)
        return os.path.getsize(dest) // 1024
    return 0


for key, cat, aid in PAPERS:
    dest = os.path.join(BASE, cat, f"{key}.pdf")
    # Re-download if missing or suspiciously small
    if os.path.exists(dest) and os.path.getsize(dest) > 500_000:
        print(f"[SKIP] {key} ({os.path.getsize(dest)//1024} KB)")
        continue
    print(f"[DL] {key}: arxiv {aid} ...", end=" ")
    sz = download(aid, dest)
    if sz:
        print(f"[OK] {sz} KB")
    else:
        print("[FAIL]")
    time.sleep(2)
