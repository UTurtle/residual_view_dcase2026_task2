"""Waveform input types for raw input-type ablations."""

from __future__ import annotations

import numpy as np

WAVEFORM_INPUT_TYPES = {
    "near",
    "far",
    "near_minus_0.5_far",
    "near_minus_0.75_far",
    "near_minus_far",
}

INPUT_TYPE_NAMES = WAVEFORM_INPUT_TYPES | {"near_with_far"}


def make_input_type(wave: np.ndarray, input_type: str) -> np.ndarray:
    """Create one mono waveform input type from 2-channel DCASE2026 audio."""
    if wave.ndim != 2 or wave.shape[1] < 2:
        raise ValueError(
            f"Input type {input_type} requires 2-channel audio, "
            f"got shape={wave.shape}."
        )
    near = wave[:, 0]
    far = wave[:, 1]
    if input_type == "near":
        return near
    if input_type == "far":
        return far
    if input_type == "near_minus_0.5_far":
        return near - 0.5 * far
    if input_type == "near_minus_0.75_far":
        return near - 0.75 * far
    if input_type == "near_minus_far":
        return near - far
    raise ValueError(
        f"Unknown waveform input_type={input_type}. "
        f"Expected one of {sorted(WAVEFORM_INPUT_TYPES)}."
    )


def expand_input_type(input_type: str, split_role: str) -> list[str]:
    """Map an input ablation condition to train/test waveform input types."""
    if input_type not in INPUT_TYPE_NAMES:
        raise ValueError(
            f"Unknown input_type={input_type}. "
            f"Expected one of {sorted(INPUT_TYPE_NAMES)}."
        )
    if split_role not in {"train", "eval"}:
        raise ValueError(
            f"Unknown split_role={split_role}. Expected 'train' or 'eval'."
        )
    if input_type == "near_with_far":
        if split_role == "train":
            return ["near", "far"]
        return ["near"]
    return [input_type]
