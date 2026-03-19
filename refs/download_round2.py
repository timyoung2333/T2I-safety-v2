#!/usr/bin/env python3
"""Round 2: download remaining papers via arXiv search API."""
import os, time, requests, xml.etree.ElementTree as ET

BASE = os.path.dirname(os.path.abspath(__file__))

# Papers that failed in round 1, with better search terms and known arXiv IDs where possible
PAPERS = [
    # ATTACKS
    ("unlearndiffatk_eccv24", "attack", "2309.11907"),  # UnlearnDiffAtk
    ("subattack_iclr26", "attack", None, "interpretable token embeddings jailbreaking diffusion unlearning"),
    ("sneakyprompt_sp24", "attack", "2305.12082"),  # SneakyPrompt
    ("jailfuzzer_sp25", "attack", None, "jailbreaking text-to-image models LLM-based agents"),
    ("surrogateprompt_ccs24", "attack", None, "SurrogatePrompt bypassing safety filter text-to-image substitution"),
    ("art_neurips24", "attack", "2405.01860"),  # ART
    ("exposingguardrails_usenix25", "attack", None, "exposing guardrails reverse engineering jailbreaking safety filters AI image"),
    ("trojanip_cvpr25", "attack", None, "trojan horse image prompt adapter jailbreaks text-to-image"),

    # DEFENSES
    ("sld_cvpr23", "defense", "2211.05105"),  # SLD
    ("safree_iclr25", "defense", "2410.12761"),  # SAFREE
    ("safedenoiser_neurips25", "defense", None, "training-free safe denoisers diffusion models"),
    ("hirm_iclr26", "defense", None, "localized concept erasure high-level representation misdirection"),
    ("continualunlearning_iclr26", "defense", None, "continual unlearning text-to-image diffusion regularization"),
    ("salun_iclr24", "defense", "2310.12508"),  # SalUn
    ("mace_cvpr24", "defense", "2403.06135"),  # MACE
    ("advunlearn_neurips24", "defense", "2407.10937"),  # AdvUnlearn
    ("gloce_cvpr25", "defense", None, "GLoCE gated low-rank concept erasure training-free diffusion"),
    ("dag_cvpr25", "defense", None, "detect guide self-regulation diffusion safe text-to-image guideline token"),
    ("safegen_ccs24", "defense", "2404.06666"),  # SafeGen
    ("safeguider_ccs25", "defense", None, "SafeGuider defending jailbreak text-to-image semantic-aware safety"),
    ("embsanitizer_arxiv24", "defense", "2404.12782"),  # Embedding Sanitizer
    ("llavaguard_icml25", "defense", "2406.05113"),  # LlavaGuard

    # BENCHMARKS
    ("t2isafety_cvpr25", "benchmark", None, "T2ISafety benchmark assessing fairness toxicity privacy image generation"),
    ("jailbreakdiffbench_iccv25", "benchmark", None, "JailbreakDiffBench comprehensive benchmark jailbreaking diffusion"),
    ("t2iriskyprompt_aaai26", "benchmark", None, "T2I-RiskyPrompt dataset benchmark risky prompt detection"),
    ("unsafebench_ccs25", "benchmark", "2405.11152"),  # UnsafeBench
]


def download_arxiv(arxiv_id, dest):
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and len(r.content) > 10000:
            with open(dest, "wb") as f:
                f.write(r.content)
            return True
    except Exception as e:
        print(f"    Download error: {e}")
    return False


def search_arxiv(query):
    """Search arXiv API and return first result's ID."""
    url = "http://export.arxiv.org/api/query"
    params = {"search_query": f"all:{query}", "max_results": 3}
    try:
        r = requests.get(url, params=params, timeout=15)
        root = ET.fromstring(r.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", ns):
            arxiv_id = entry.find("atom:id", ns).text.split("/abs/")[-1]
            title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
            return arxiv_id, title
    except Exception as e:
        print(f"    arXiv search error: {e}")
    return None, None


def main():
    success, fail = 0, 0
    for item in PAPERS:
        if len(item) == 3:
            key, cat, arxiv_id = item
            query = None
        else:
            key, cat, arxiv_id, query = item

        dest = os.path.join(BASE, cat, f"{key}.pdf")
        if os.path.exists(dest) and os.path.getsize(dest) > 10000:
            print(f"[SKIP] {key}")
            success += 1
            continue

        if arxiv_id:
            print(f"[ARXIV] {key}: {arxiv_id}")
            if download_arxiv(arxiv_id, dest):
                size_kb = os.path.getsize(dest) // 1024
                print(f"  [OK] {size_kb} KB")
                success += 1
            else:
                print(f"  [FAIL]")
                fail += 1
        elif query:
            print(f"[SEARCH] {key}: {query[:60]}...")
            arxiv_id, title = search_arxiv(query)
            if arxiv_id:
                print(f"  Found: {arxiv_id} - {title[:60]}")
                if download_arxiv(arxiv_id, dest):
                    size_kb = os.path.getsize(dest) // 1024
                    print(f"  [OK] {size_kb} KB")
                    success += 1
                else:
                    print(f"  [FAIL] Download error")
                    fail += 1
            else:
                print(f"  [FAIL] Not found on arXiv")
                fail += 1

        time.sleep(2)

    print(f"\nRound 2 done: {success} OK, {fail} failed")


if __name__ == "__main__":
    main()
