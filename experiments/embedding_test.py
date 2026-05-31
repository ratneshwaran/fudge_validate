"""
METHODOLOGY.md TODO 1 — Embedding test.

Gate test for FuDGE viability on TV. The cost function compares an utterance
embedding to a DAG-node-label embedding. If abstract labels and real utterances
live in different regions of embedding space, FuDGE measures the wrong thing.

For each of 3 PE-style node labels:
  - embed the label string with SBERT (all-MiniLM-L6-v2)
  - embed a hand-picked set of real TV utterances that demonstrate that intent (positives)
  - embed a hand-picked set of unrelated TV utterances (negatives)
  - report mean cosine distance label->pos vs label->neg

Pass criterion: mean(pos) + 0.2 < mean(neg) for all 3 labels (consistent gap).

Run:
  python experiments/embedding_test.py
"""
import json
import sys
from pathlib import Path

sys.modules["tensorflow"] = None

import numpy as np

from fudge.embeddings import EmbeddingCache


# Three candidate PE node labels (therapist intent), one per phase region.
LABELS = {
    "orient_to_imaginal_exposure": (
        "Therapist orients the client to the imaginal exposure procedure"
    ),
    "elicit_suds_rating": (
        "Therapist elicits the client's SUDS rating during exposure"
    ),
    "process_thoughts_after_exposure": (
        "Therapist processes the client's thoughts and feelings after exposure"
    ),
}


# Hand-picked from data/thousand-voices-trauma/.../conversations/1_P{5,6,11}_conversation.json
# Positives are therapist turns that demonstrate the intent; negatives are
# therapist turns from a clearly different intent (other PE phases or other moves).
EXAMPLES = {
    "orient_to_imaginal_exposure": {
        "positives": [
            "Mr. Lee, today we'll be starting imaginal exposure. How are you feeling about that?",
            "We'll go at your pace. You'll describe the event in present tense. I'll guide you throughout. How does that sound?",
            "This exercise is designed to help you process the trauma. What questions do you have?",
            "You may feel strong emotions or physical sensations. This is normal. We'll work through it together. How do you feel about that?",
            "Absolutely. You're in control. We can pause anytime. Shall we start with a breathing exercise?",
        ],
        "negatives": [
            "What's your SUDS rating now?",
            "It's common to feel that way. How might this connect to your day-to-day life?",
            "You should be proud. Let's end the exercise here. How are you feeling overall?",
            "Those sound like excellent steps. How do you feel about our session today?",
            "Thank you for sharing that. What do you think about these realizations?",
        ],
    },
    "elicit_suds_rating": {
        "positives": [
            "You're doing well. What's your SUDS now?",
            "How are you feeling right now? What's your SUDS?",
            "What's your SUDS rating now?",
            "How are you feeling now? What's your SUDS?",
            "That's great progress. How would you rate your SUDS now?",
        ],
        "negatives": [
            "Mr. Lee, today we'll be starting imaginal exposure. How are you feeling about that?",
            "How did that feel, physically?",
            "That's a great goal. What's one small step you could take this week?",
            "You may feel strong emotions or physical sensations. This is normal. We'll work through it together. How do you feel about that?",
            "Your dedication is commendable. Let's begin with deep breaths. In through your nose, out through your mouth.",
        ],
    },
    "process_thoughts_after_exposure": {
        "positives": [
            "How are you feeling after that imaginal exposure, Mr. Lee?",
            "That's understandable. What stood out to you most?",
            "It's common to feel that way. How might this connect to your day-to-day life?",
            "Thank you for sharing that. What do you think about these realizations?",
            "That's a positive perspective. How do you feel about continuing this work?",
        ],
        "negatives": [
            "What's your SUDS now?",
            "We'll go at your pace. You'll describe the event in present tense. I'll guide you throughout. How does that sound?",
            "Let's begin our imaginal exposure exercise. Can you describe the airplane crash scene and give me your initial SUDS rating?",
            "Your dedication is commendable. Let's begin with deep breaths. In through your nose, out through your mouth.",
            "Excellent. Remember, you're safe here. Describe where you were before boarding the plane.",
        ],
    },
}


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine similarity; embeddings are pre-normalised by EmbeddingCache."""
    return float(1.0 - np.dot(a, b))


def run() -> dict:
    cache = EmbeddingCache()
    results: dict = {"labels": {}, "summary": {}}
    all_pass = True

    for key, label_text in LABELS.items():
        label_vec = cache.encode(label_text)
        pos_texts = EXAMPLES[key]["positives"]
        neg_texts = EXAMPLES[key]["negatives"]
        pos_vecs = cache.encode_batch(pos_texts)
        neg_vecs = cache.encode_batch(neg_texts)

        pos_dists = [cosine_distance(label_vec, v) for v in pos_vecs]
        neg_dists = [cosine_distance(label_vec, v) for v in neg_vecs]

        pos_mean = float(np.mean(pos_dists))
        pos_std = float(np.std(pos_dists))
        neg_mean = float(np.mean(neg_dists))
        neg_std = float(np.std(neg_dists))
        gap = neg_mean - pos_mean
        passed = gap >= 0.2

        if not passed:
            all_pass = False

        print(f"\n[{key}]  label = {label_text!r}")
        print(f"  positives: mean={pos_mean:.3f} +/- {pos_std:.3f}  (n={len(pos_dists)})")
        for t, d in zip(pos_texts, pos_dists):
            print(f"    {d:.3f}  {t}")
        print(f"  negatives: mean={neg_mean:.3f} +/- {neg_std:.3f}  (n={len(neg_dists)})")
        for t, d in zip(neg_texts, neg_dists):
            print(f"    {d:.3f}  {t}")
        print(f"  gap (neg-pos) = {gap:+.3f}   {'PASS' if passed else 'FAIL'}")

        results["labels"][key] = {
            "label": label_text,
            "positives": {
                "texts": pos_texts,
                "distances": pos_dists,
                "mean": pos_mean,
                "std": pos_std,
            },
            "negatives": {
                "texts": neg_texts,
                "distances": neg_dists,
                "mean": neg_mean,
                "std": neg_std,
            },
            "gap": gap,
            "passed": passed,
        }

    results["summary"] = {
        "all_pass": all_pass,
        "criterion": "neg_mean - pos_mean >= 0.20 for every label",
        "verdict": (
            "FuDGE viable on TV — abstract labels separate similar from "
            "unrelated TV utterances."
            if all_pass else
            "FuDGE NOT viable on TV — abstract labels do not separate. "
            "Fall back to AutoEval-ToD Domain Compliance only."
        ),
    }

    out_dir = Path("experiments/embedding_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Summary ===")
    print(results["summary"]["verdict"])
    print(f"Wrote {out_path}")
    return results


if __name__ == "__main__":
    run()
