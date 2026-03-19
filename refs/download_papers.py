#!/usr/bin/env python3
"""Download T2I safety papers organized by category."""
import os, time, requests, sys

BASE = os.path.dirname(os.path.abspath(__file__))

# (citation_key, category, search_title)
PAPERS = [
    # === ATTACKS ===
    ("ringabell_iclr24", "attack", "Ring-A-Bell How Reliable are Concept Removal Methods for Diffusion Models"),
    ("mma_cvpr24", "attack", "MMA-Diffusion MultiModal Attack on Diffusion Models"),
    ("p4d_icml24", "attack", "Prompting4Debugging Red-Teaming Text-to-Image Diffusion Models by Finding Problematic Prompts"),
    ("unlearndiffatk_eccv24", "attack", "To Generate or Not Safety-Driven Unlearned Diffusion Models Are Still Easy To Generate Unsafe Images"),
    ("subattack_iclr26", "attack", "Dual Power of Interpretable Token Embeddings Jailbreaking Attacks and Defenses for Diffusion Model Unlearning"),
    ("sneakyprompt_sp24", "attack", "SneakyPrompt Jailbreaking Text-to-Image Generative Models"),
    ("jailfuzzer_sp25", "attack", "JailBreaking Text-to-Image Models with LLM-Based Agents"),
    ("surrogateprompt_ccs24", "attack", "SurrogatePrompt Bypassing the Safety Filter of Text-to-Image Models via Substitution"),
    ("pgj_aaai25", "attack", "Perception-Guided Jailbreak Text-to-Image Models"),
    ("art_neurips24", "attack", "ART Automatic Red-Teaming for Text-to-Image Models"),
    ("exposingguardrails_usenix25", "attack", "Exposing the Guardrails Reverse Engineering and Jailbreaking Safety Filters of AI Image Generators"),
    ("trojanip_cvpr25", "attack", "Mind the Trojan Horse Image Prompt Adapter Enabling Scalable and Adaptive Jailbreaks"),

    # === DEFENSES ===
    ("sld_cvpr23", "defense", "Safe Latent Diffusion Mitigating Inappropriate Degeneration in Diffusion Models"),
    ("safree_iclr25", "defense", "SAFREE Training-Free and Adaptive Guard for Safe Text-to-Image"),
    ("stg_neurips25", "defense", "Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models"),
    ("safedenoiser_neurips25", "defense", "Training-Free Safe Denoisers for Safe Use of Diffusion Models"),
    ("speed_iclr26", "defense", "SPEED Scalable Precise and Efficient Concept Erasure for Diffusion Models"),
    ("hirm_iclr26", "defense", "Localized Concept Erasure Text-to-Image Diffusion Models High-Level Representation Misdirection"),
    ("continualunlearning_iclr26", "defense", "Continual Unlearning for Text-to-Image Diffusion Models Regularization Perspective"),
    ("esd_iccv23", "defense", "Erasing Concepts from Diffusion Models"),
    ("uce_wacv24", "defense", "Unified Concept Editing in Diffusion Models"),
    ("salun_iclr24", "defense", "SalUn Empowering Machine Unlearning via Gradient-based Weight Saliency"),
    ("mace_cvpr24", "defense", "MACE Mass Concept Erasure in Diffusion Models"),
    ("advunlearn_neurips24", "defense", "Defensive Unlearning with Adversarial Training for Robust Concept Erasure in Diffusion Models"),
    ("gloce_cvpr25", "defense", "GLoCE Gated Low-rank Concept Erasure Training-Free Diffusion Model Safety"),
    ("dag_cvpr25", "defense", "Detect-and-Guide Self-regulation Diffusion Models Safe Text-to-Image Generation Guideline Token"),
    ("safegen_ccs24", "defense", "SafeGen Mitigating Sexually Explicit Content Generation in Text-to-Image Models"),
    ("safeclip_eccv24", "defense", "Safe-CLIP Removing NSFW Concepts from Vision-and-Language Models"),
    ("safeguider_ccs25", "defense", "SafeGuider Defending Against Jailbreak Attacks Text-to-Image Semantic-Aware Safety Guidance"),
    ("embsanitizer_arxiv24", "defense", "Embedding Sanitizer Vision-agnostic Plug-and-play Safety Module Text-to-Image"),
    ("llavaguard_icml25", "defense", "LlavaGuard VLM-based Safeguard for Vision Dataset Curation and Safety Assessment"),

    # === BENCHMARKS ===
    ("implicitbench_icml24", "benchmark", "Towards Implicit Prompt for Text-to-Image Models"),
    ("copro_eccv24", "benchmark", "Latent Guard Safety Framework for Text-to-Image Generation"),
    ("t2isafety_cvpr25", "benchmark", "T2ISafety Benchmark for Assessing Fairness Toxicity and Privacy in Image Generation"),
    ("jailbreakdiffbench_iccv25", "benchmark", "JailbreakDiffBench Comprehensive Benchmark for Jailbreaking Diffusion Models"),
    ("siuo_naacl25", "benchmark", "SIUO Safe Inputs Unsafe Output Benchmarking Cross-modality Safety"),
    ("t2iriskyprompt_aaai26", "benchmark", "T2I-RiskyPrompt Dataset and Benchmark for Risky Prompt Detection"),
    ("unsafebench_ccs25", "benchmark", "UnsafeBench Benchmarking Image Safety Classifiers on Real-World and AI-Generated Images"),
]


def search_semantic_scholar(title):
    """Search Semantic Scholar for a paper and return its open access PDF URL."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": title, "limit": 3, "fields": "title,openAccessPdf,externalIds"}
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            data = r.json()
            for paper in data.get("data", []):
                pdf_info = paper.get("openAccessPdf")
                if pdf_info and pdf_info.get("url"):
                    return pdf_info["url"]
                # Try arXiv fallback
                ext = paper.get("externalIds", {})
                if ext and ext.get("ArXiv"):
                    return f"https://arxiv.org/pdf/{ext['ArXiv']}.pdf"
    except Exception as e:
        print(f"    Search error: {e}")
    return None


def download_pdf(url, path):
    """Download a PDF from URL to path."""
    try:
        r = requests.get(url, timeout=30, stream=True)
        if r.status_code == 200 and len(r.content) > 10000:
            with open(path, "wb") as f:
                f.write(r.content)
            return True
    except Exception as e:
        print(f"    Download error: {e}")
    return False


def main():
    success, fail = 0, 0
    for key, cat, title in PAPERS:
        dest = os.path.join(BASE, cat, f"{key}.pdf")
        if os.path.exists(dest) and os.path.getsize(dest) > 10000:
            print(f"[SKIP] {key} (already exists)")
            success += 1
            continue

        print(f"[SEARCH] {key}: {title[:60]}...")
        pdf_url = search_semantic_scholar(title)

        if pdf_url:
            print(f"  Found: {pdf_url[:80]}")
            if download_pdf(pdf_url, dest):
                size_kb = os.path.getsize(dest) // 1024
                print(f"  [OK] Saved ({size_kb} KB)")
                success += 1
            else:
                print(f"  [FAIL] Download failed")
                fail += 1
        else:
            print(f"  [FAIL] No open access PDF found")
            fail += 1

        time.sleep(1.5)  # Rate limit

    print(f"\nDone: {success} downloaded, {fail} failed")


if __name__ == "__main__":
    main()
