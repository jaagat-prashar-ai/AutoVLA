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


def prefix_truncate(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def suffix_truncate(text: str, n: int) -> str:
    if n <= 0:
        return ""
    return " ".join(text.split()[-n:])


def inject_mistakes(text: str) -> Tuple[str, List[str]]:
    fired: List[str] = []

    def _sub(m: "re.Match") -> str:
        key = m.group(0).lower()
        fired.append(key)
        return MISTAKE_SWAPS[key]

    edited = _MISTAKE_PATTERN.sub(_sub, text)
    return edited, fired


# ---------------------------------------------------------------------------
# Trajectory metrics
# ---------------------------------------------------------------------------

def trajectory_deltas(baseline: np.ndarray, other: np.ndarray) -> dict:
    T = min(len(baseline), len(other))
    if T == 0:
        return {"ade_m": None, "endpoint_m": None, "delta_xy_per_waypoint": []}
    delta_xy = np.linalg.norm(other[:T, :2] - baseline[:T, :2], axis=-1)
    return {
        "ade_m": float(delta_xy.mean()),
        "endpoint_m": float(delta_xy[-1]),
        "delta_xy_per_waypoint": delta_xy.round(4).tolist(),
    }


def _heading_rate(traj: np.ndarray, dt: float) -> np.ndarray:
    dh = np.diff(traj[:, 2])
    dh = (dh + np.pi) % (2 * np.pi) - np.pi
    return dh / dt


def _speed(traj: np.ndarray, dt: float) -> np.ndarray:
    return np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=-1) / dt


def control_deltas(baseline: np.ndarray, other: np.ndarray, dt: float) -> dict:
    if len(baseline) < 2 or len(other) < 2:
        return {"d_curvature_mean": None, "d_accel_mean": None}
    hb, ho = _heading_rate(baseline, dt), _heading_rate(other, dt)
    sb, so = _speed(baseline, dt), _speed(other, dt)
    Th = min(len(hb), len(ho))
    d_curv = np.abs(ho[:Th] - hb[:Th]) if Th > 0 else np.array([0.0])

    ab = np.diff(sb) / dt if len(sb) > 1 else np.array([])
    ao = np.diff(so) / dt if len(so) > 1 else np.array([])
    Ta = min(len(ab), len(ao))
    d_accel = np.abs(ao[:Ta] - ab[:Ta]) if Ta > 0 else np.array([0.0])

    return {
        "d_curvature_mean": float(d_curv.mean()),
        "d_accel_mean": float(d_accel.mean()),
    }


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _to_device(inputs, device: str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}


def generate_full(autovla: AutoVLA, input_features: dict, seed: int, device: str) -> Optional[dict]:
    """Full from-scratch generation (prompt -> reasoning text + action tokens),
    following the same call pattern as AutoVLA.predict(), but also returning the
    raw reasoning/action token split so callers can edit the reasoning span."""
    prompt_inputs = autovla.get_prompt(input_features)
    model_inputs = _to_device(prompt_inputs, device)

    torch.manual_seed(seed)
    with torch.no_grad():
        generated = autovla.vlm.generate(
            **model_inputs,
            max_length=autovla.gen_conf["max_length"],
            do_sample=True,
            temperature=autovla.gen_conf["temperature"],
            top_k=autovla.gen_conf["top_k"],
            top_p=autovla.gen_conf["top_p"],
        )

    prompt_len = model_inputs["input_ids"].shape[1]
    completion = generated[0, prompt_len:][:-1].cpu()  # drop trailing eos, mirrors AutoVLA.predict()

    action_mask = completion >= autovla.action_start_id
    if not action_mask.any():
        return None
    action_start_idx = int(action_mask.nonzero()[0].item())
    reasoning_ids = completion[:action_start_idx]
    action_ids = completion[action_start_idx:]

    trajectory = autovla.action_tokenizer.decode_token_ids_to_trajectory(action_ids)
    if len(trajectory) == 0:
        return None

    return {
        "model_inputs": model_inputs,
        "reasoning_text": autovla.processor.decode(reasoning_ids).strip(),
        "trajectory": trajectory[0, 1:].numpy(),
    }


def continue_from_text(
    autovla: AutoVLA,
    prompt_model_inputs: Dict[str, torch.Tensor],
    edited_text: str,
    max_new_tokens: int,
    seed: int,
) -> Optional[np.ndarray]:
    """Teacher-force `edited_text` as the assistant's reasoning-so-far by
    appending its tokens after the prompt, then let the model continue
    autoregressively (fresh words + action tokens) from there."""
    if edited_text.strip():
        edited_ids = autovla.processor.tokenizer(
            edited_text, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(prompt_model_inputs["input_ids"].device)
    else:
        edited_ids = torch.empty(
            (1, 0), dtype=prompt_model_inputs["input_ids"].dtype,
            device=prompt_model_inputs["input_ids"].device,
        )

    forced_input_ids = torch.cat([prompt_model_inputs["input_ids"], edited_ids], dim=1)
    forced_attention_mask = torch.cat(
        [prompt_model_inputs["attention_mask"], torch.ones_like(edited_ids)], dim=1
    )

    torch.manual_seed(seed)
    with torch.no_grad():
        generated = autovla.vlm.generate(
            input_ids=forced_input_ids,
            attention_mask=forced_attention_mask,
            pixel_values_videos=prompt_model_inputs["pixel_values_videos"],
            video_grid_thw=prompt_model_inputs["video_grid_thw"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=autovla.gen_conf["temperature"],
            top_k=autovla.gen_conf["top_k"],
            top_p=autovla.gen_conf["top_p"],
        )

    continuation = generated[0, forced_input_ids.shape[1]:].cpu()
    action_ids = continuation[continuation >= autovla.action_start_id]
    if len(action_ids) == 0:
        return None

    trajectory = autovla.action_tokenizer.decode_token_ids_to_trajectory(action_ids)
    if len(trajectory) == 0:
        return None
    return trajectory[0, 1:].numpy()
