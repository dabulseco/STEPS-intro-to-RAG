#!/usr/bin/env python3
"""
Retrieval evaluation harness.

Answers the question every toggle in this app raises: *did that actually help?*
Retrieval quality cannot be judged by reading one answer -- a change that fixes
the question you happen to be testing often breaks two you are not. This runs a
fixed set of questions against a fixed corpus under several configurations and
reports hit rate and MRR for each.

It measures **retrieval**, not answer quality: whether the passage containing
the answer was found, not whether the model wrote a good paragraph. That is the
half the settings here actually control, and it needs no model call to score.

Usage:
    python tools/eval_retrieval.py --docs path/to/docs --questions golden.json
    python tools/eval_retrieval.py --demo        # synthetic corpus, no files needed

The golden file is a JSON list of:
    [{"question": "...", "expect": "substring that must appear in a hit"}, ...]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_core.chunker import chunk_document, chunk_documents
from rag_core.embeddings import OllamaEmbedder
from rag_core.pipeline import RetrievalSettings, retrieve
from rag_core.store import EphemeralRagStore
from utils.document_processor import DocumentProcessor

# Configurations to compare. Each is a partial RetrievalSettings; the harness
# holds everything else constant so differences are attributable.
CONFIGURATIONS = {
    "semantic only":   {"dense_weight": 1.0},
    "keyword only":    {"dense_weight": 0.0},
    "hybrid (0.5)":    {"dense_weight": 0.5},
    "hybrid + more k": {"dense_weight": 0.5, "dense_k": 40, "sparse_k": 40, "fused_k": 20},
}


class LocalFile:
    """Adapts a path to the interface DocumentProcessor expects."""

    def __init__(self, path: Path):
        self._data = path.read_bytes()
        self.name = path.name
        self.size = len(self._data)

    def read(self) -> bytes:
        return self._data


def build_demo_corpus():
    """
    A synthetic corpus built to *separate* the configurations rather than
    flatter them.

    Two properties make it discriminating. First, every target fact is
    surrounded by near-identical distractors differing only in the value, so
    finding the right one demands precision rather than topical similarity.
    Second, the questions are split between paraphrases sharing no vocabulary
    with the source (which keyword search cannot reach) and bare identifiers
    carrying no meaning (which semantic search handles poorly).

    A corpus where every configuration scores 100% teaches nothing.
    """
    targets = [
        ("The calibration constant for the flow sensor is 0.0472 volts per liter.", "0.0472"),
        ("Cells in the treatment arm were cultured at 37 degrees Celsius in DMEM medium.", "37 degrees"),
        ("Whole-genome sequencing used the Illumina NovaSeq 6000 platform.", "NovaSeq 6000"),
        ("The CFTR-KO line showed chloride transport of 38 nanoamps per square centimetre.", "38 nanoamps"),
        ("Reagents for the second cohort were supplied under catalog number ZZQX7-4419.", "ZZQX7-4419"),
    ]

    # Near-misses: same topic and phrasing, different subject and value.
    distractors = [
        "The calibration constant for the pressure sensor is 0.0918 volts per bar.",
        "The calibration constant for the temperature probe is 0.0231 volts per kelvin.",
        "Cells in the control arm were cultured at 30 degrees Celsius in RPMI medium.",
        "Cells in the pilot study were cultured at 25 degrees Celsius in minimal medium.",
        "Targeted sequencing used the Illumina MiSeq platform.",
        "Long-read sequencing used the Oxford Nanopore PromethION platform.",
        "The wildtype line showed chloride transport of 100 nanoamps per square centimetre.",
        "The heterozygous line showed chloride transport of 71 nanoamps per square centimetre.",
        "Reagents for the first cohort were supplied under catalog number ZZQX7-1180.",
        "Reagents for the pilot were supplied under catalog number PLQR2-8890.",
    ]

    blocks = []
    for i in range(60):
        blocks.append({
            "text": f"Routine observations for interval {i} were recorded without incident. "
                    f"Instrument drift during interval {i} remained within tolerance.",
            "page": i + 1, "slide": None, "section": f"Section {i}", "block_index": i,
        })
    for j, text in enumerate(distractors):
        blocks.append({"text": text, "page": 80 + j, "slide": None,
                       "section": "Supplementary", "block_index": 60 + j})
    for j, (text, _) in enumerate(targets):
        blocks.append({"text": text, "page": 100 + j, "slide": None,
                       "section": "Appendix", "block_index": 70 + j})

    questions = [
        # Bare identifiers: no semantic content, pure lexical match.
        {"question": "ZZQX7-4419", "expect": "ZZQX7-4419"},
        {"question": "NovaSeq 6000", "expect": "NovaSeq 6000"},
        # Paraphrases: no shared vocabulary with the source sentence.
        {"question": "how warm were the treated cells kept while growing",
         "expect": "37 degrees"},
        {"question": "what happens to ion movement when the gene is knocked out",
         "expect": "38 nanoamps"},
        # Specific questions among near-identical neighbours.
        {"question": "What is the calibration constant for the flow sensor?",
         "expect": "0.0472"},
        {"question": "Which catalog number covered the second cohort reagents?",
         "expect": "ZZQX7-4419"},
    ]
    return chunk_document(blocks, "demo.pdf", ".pdf"), questions


def load_documents(docs_dir: Path):
    """Extract and chunk every supported document in a directory."""
    paths = [
        p for p in sorted(docs_dir.iterdir())
        if p.is_file() and DocumentProcessor.is_supported_file(p.name)
    ]
    if not paths:
        raise SystemExit(f"No supported documents found in {docs_dir}")

    results = DocumentProcessor.process_multiple_documents([LocalFile(p) for p in paths])
    for result in results:
        status = "ERROR: " + result["error"] if result["error"] else f"{len(result['blocks'])} blocks"
        print(f"  {result['filename']:40s} {status}")

    return chunk_documents(results)


def evaluate(store, questions: Sequence[Dict], settings: RetrievalSettings) -> Dict:
    """
    Score one configuration.

    Returns:
        hit_rate (found anywhere in top-k), mrr (rank-sensitive), and latency
    """
    hits = 0
    reciprocal_ranks = 0.0
    misses = []
    started = time.time()

    for item in questions:
        results = retrieve(store, item["question"], settings)
        expected = item["expect"].lower()

        rank = next(
            (i for i, r in enumerate(results) if expected in r.chunk.text.lower()),
            None,
        )
        if rank is None:
            misses.append(item["question"])
        else:
            hits += 1
            reciprocal_ranks += 1.0 / (rank + 1)

    total = len(questions) or 1
    return {
        "hit_rate": hits / total,
        "mrr": reciprocal_ranks / total,
        "seconds": time.time() - started,
        "misses": misses,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=Path, help="Directory of PDF/DOCX/PPTX files")
    parser.add_argument("--questions", type=Path, help="JSON file of golden questions")
    parser.add_argument("--demo", action="store_true", help="Run on a synthetic corpus")
    parser.add_argument("--embed-model", default="nomic-embed-text",
                        help="Ollama embedding model (default: nomic-embed-text)")
    parser.add_argument("--top-k", type=int, default=5, help="Passages scored as hits")
    args = parser.parse_args()

    if args.demo:
        print("Building demo corpus...")
        chunks, questions = build_demo_corpus()
    else:
        if not args.docs or not args.questions:
            parser.error("--docs and --questions are required unless --demo is given")
        print(f"Loading documents from {args.docs}...")
        chunks = load_documents(args.docs)
        questions = json.loads(args.questions.read_text())

    print(f"\n{len(chunks)} chunks, {len(questions)} questions")
    print(f"Embedding with {args.embed_model}...")

    store = EphemeralRagStore(OllamaEmbedder(model=args.embed_model))
    store.add_chunks(chunks)

    print(f"\n{'configuration':20s} {'hit rate':>10s} {'MRR':>8s} {'seconds':>9s}")
    print("-" * 50)

    outcomes = {}
    try:
        for name, overrides in CONFIGURATIONS.items():
            settings = RetrievalSettings(final_k=args.top_k, **overrides)
            outcome = evaluate(store, questions, settings)
            outcomes[name] = outcome
            print(f"{name:20s} {outcome['hit_rate']:>9.0%} {outcome['mrr']:>8.3f} "
                  f"{outcome['seconds']:>9.1f}")

        print("\nQuestions missed, by configuration:")
        for name, outcome in outcomes.items():
            if outcome["misses"]:
                print(f"  {name}:")
                for miss in outcome["misses"]:
                    print(f"    - {miss}")
            else:
                print(f"  {name}: none")

        best = max(outcomes.items(), key=lambda pair: (pair[1]["mrr"], pair[1]["hit_rate"]))
        print(f"\nBest by MRR: {best[0]}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
