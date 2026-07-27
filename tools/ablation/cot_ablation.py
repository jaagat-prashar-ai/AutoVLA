"""
cot_ablation.py — Text-level CoT ablation probes for AutoVLA.

AutoVLA's chain-of-thought is plain generated text sharing one autoregressive
stream with its physical action tokens (see models/autovla.py:AutoVLA.get_prompt
and models/action_tokenizer.py), not a separately-addressable set of hidden
states. That's different from the masking/ subsystem's MaskedAlpamayo1_5, which
exposes custom primitives (compare_conditions, salience_leave_one_word_out) that
mask specific attention columns in a single forward pass across several
conditions at once. AutoVLA has no equivalent, so the ablations here work at the
text/token level instead:

  1. Generate one baseline rollout per scene and split its completion into a
     reasoning-text span and an action-token span (the point where token ids
     first cross `action_start_id`).
  2. Edit the reasoning text for a given condition (mask concept words, keep
     only a prefix/suffix, or inject a plausible mistake).
  3. Teacher-force the edited text back in as the assistant's "reasoning so
     far" (tokenized and appended after the prompt) and let the model continue
     generating autoregressively -- its own words plus fresh action tokens --
     from that point.
  4. Decode the resulting action tokens to a trajectory and compare it to the
     baseline trajectory.

This measures how much the *decoded trajectory* actually moves in response to
what the model is shown as its own prior reasoning -- a faithfulness probe, not
just a text-quality check.

Conditions (adapted from masking/training/run.py's experiments a-d):
  no_cot          -- experiment A: reasoning vs. no-reasoning, by toggling
                     `use_cot` and regenerating the full response from scratch.
  concept_mask    -- experiment B: strip concept-relevant words (e.g.
                     "pedestrian", "stop", "red") from the reasoning text.
  prefix_{n}w /
  suffix_{n}w     -- experiment C: keep only the first/last n words of the
                     reasoning text, sweeping n over --threshold_words.
  injected_mistake -- in the spirit of experiment D's clause-reversal probe:
                     swap decision-relevant words for plausible-but-wrong
                     opposites (stop<->accelerate, red<->green, left<->right,
                     pedestrian->no pedestrian, ...) and see whether the
                     trajectory follows the injected error.

Metrics per condition, relative to the baseline trajectory (mirrors masking's
ade_m/endpoint_m/curvature/accel fields, computed here via simple finite
differences since AutoVLA's action_tokenizer doesn't expose a `controls` dict):
  ade_m, endpoint_m, delta_xy_per_waypoint, d_curvature_mean, d_accel_mean

This is intentionally a single-process CLI script, matching tools/eval/
nusc_eval.py's loading/eval convention -- it is not a multi-GPU/Lilypad
launcher. Requires an SFT (or CoT-capable) checkpoint and a preprocessed
val split, same prerequisites as nusc_eval.py.

Usage:
    python tools/ablation/cot_ablation.py \
        --config config/training/qwen2.5-vl-3B-mix-sft.yaml \
        --checkpoint /path/to/sft_checkpoint.ckpt \
        --num_samples 50 \
        --output ablation_results.jsonl
"""
import argparse
import json
import logging
import re
import string
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "navsim"))

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoProcessor

from dataset_utils.sft_dataset import SFTDataset
from models.autovla import SFTAutoVLA, AutoVLA

logger = logging.getLogger(__name__)

DEFAULT_CONCEPTS = "pedestrian,person,cyclist,crosswalk,vehicle,stop,red,light"
DEFAULT_THRESHOLD_WORDS = "0,5,10,20,30,50"

# Word/phrase swaps used by the injected_mistake condition: each key, if found
# in the reasoning text (case-insensitive, whole word/phrase), is replaced by
# its value to inject a plausible-but-wrong claim or decision.
MISTAKE_SWAPS: Dict[str, str] = {
    "turn left": "turn right",
    "turn right": "turn left",
    "change lane to left": "change lane to right",
    "change lane to right": "change lane to left",
    "quick acceleration": "quick deceleration",
    "quick deceleration": "quick acceleration",
    "acceleration": "deceleration",
    "deceleration": "acceleration",
    "stop": "accelerate",
    "red": "green",
    "green": "red",
    "pedestrian": "no pedestrian",
    "crossing": "clear",
}
_MISTAKE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(MISTAKE_SWAPS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="CoT ablation probes for AutoVLA")
    parser.add_argument("--config", type=str, required=True, help="Path to the training/eval config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--output", type=str, default="ablation_results.jsonl")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of val scenes to run (default: all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concepts", type=str, default=DEFAULT_CONCEPTS,
                         help="Comma-separated concept words for the concept_mask condition")
    parser.add_argument("--threshold_words", type=str, default=DEFAULT_THRESHOLD_WORDS,
                         help="Comma-separated word-count thresholds for the prefix/suffix sweep")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                         help="Continuation budget for teacher-forced conditions")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Text-editing helpers for each condition
# ---------------------------------------------------------------------------

def concept_mask(text: str, concepts: List[str]) -> Tuple[str, int]:
    """Drop any word that starts with one of `concepts` (case-insensitive,
    so plural/inflected forms like "pedestrians" are also caught)."""
    concept_list = [c.strip().lower() for c in concepts if c.strip()]

    def is_concept(word: str) -> bool:
        bare = word.strip(string.punctuation).lower()
        return any(bare.startswith(c) for c in concept_list if bare)

    words = text.split()
    kept = [w for w in words if not is_concept(w)]
    return " ".join(kept), len(words) - len(kept)
