"""Embedding-space different-view ablations for near/far features.

The filename keeps the user's requested spelling: `differet_view.py`.
"""

from __future__ import annotations

from typing import Sequence, TypeVar

import torch

T = TypeVar("T")

DIFFERENT_VIEW_NAMES = {
    "fixed_residual_view",
    "near_minus_0.5_far",
    "near_minus_near_mul_far",
    "near_minus_projected_far",
    "near_minus_normalized_far",
    "joint_pair_norm_residual",
    "joint_pair_sum_norm_residual",
    "near_with_far",
    "near_mix_far",
    "near_div_far",
    "near_dot_far",
    "near_far_qk_scaled_dot",
    "near_mul_far",
    "near_plus_far",
    "near_far_concat_cross_mul",
    "near_far_concat_cross_mul_rp",
}

SCALED_DIFFERENT_VIEW_NAMES = {
    "fixed_residual_view",
    "near_minus_near_mul_far",
    "near_minus_projected_far",
    "near_minus_normalized_far",
    "joint_pair_norm_residual",
    "joint_pair_sum_norm_residual",
}

PAIR_INPUT_TYPES = ["near", "far"]


def deterministic_random_projection(
    values: torch.Tensor,
    output_dim: int,
    seed: int = 0,
) -> torch.Tensor:
    """Project the last dimension with a fixed random Gaussian matrix."""
    input_dim = values.shape[-1]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn(
        input_dim,
        output_dim,
        generator=generator,
        dtype=torch.float32,
    )
    projection = projection / (input_dim ** 0.5)
    projection = projection.to(device=values.device, dtype=values.dtype)
    return values @ projection


def _safe_l2_normalize(values: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(values.dtype).eps if values.is_floating_point() else 1e-12
    norm = values.norm(dim=-1, keepdim=True).clamp_min(eps)
    return values / norm


def _project_near_onto_far(near: torch.Tensor, far: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(far.dtype).eps if far.is_floating_point() else 1e-12
    numerator = (near * far).sum(dim=-1, keepdim=True)
    denominator = (far * far).sum(dim=-1, keepdim=True).clamp_min(eps)
    return (numerator / denominator) * far


def _joint_pair_rms_scale(near: torch.Tensor, far: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(near.dtype).eps if near.is_floating_point() else 1e-12
    near_energy = near.pow(2).sum(dim=-1, keepdim=True)
    far_energy = far.pow(2).sum(dim=-1, keepdim=True)
    return ((near_energy + far_energy) / 2.0).sqrt().clamp_min(eps)


def _joint_pair_sum_scale(near: torch.Tensor, far: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(near.dtype).eps if near.is_floating_point() else 1e-12
    return (near.norm(dim=-1, keepdim=True) + far.norm(dim=-1, keepdim=True)).clamp_min(eps)


def make_different_view_features(
    feature_layers: torch.Tensor,
    different_view: str,
    fixed_residual_alpha: float | None = None,
) -> torch.Tensor:
    """Create one embedding-space view from alternating near/far features.

    Expected shape is `(layers, 2 * n_clips, dim)`, where each clip appears as
    `near` then `far`.
    """
    if different_view not in DIFFERENT_VIEW_NAMES:
        raise ValueError(
            f"Unknown different_view={different_view}. "
            f"Expected one of {sorted(DIFFERENT_VIEW_NAMES)}."
        )
    if feature_layers.ndim != 3:
        raise ValueError(
            "different_view expects flattened layer features with shape "
            f"(layers, samples, dim), got shape={tuple(feature_layers.shape)}."
        )
    if feature_layers.shape[1] % 2 != 0:
        raise ValueError(
            "different_view expects paired near/far samples, got odd sample "
            f"count={feature_layers.shape[1]}."
        )

    near = feature_layers[:, 0::2]
    far = feature_layers[:, 1::2]

    if different_view == "fixed_residual_view":
        if fixed_residual_alpha is None:
            raise ValueError(
                "fixed_residual_view requires fixed_residual_alpha."
            )
        return near - fixed_residual_alpha * far
    if different_view == "near_minus_near_mul_far":
        if fixed_residual_alpha is None:
            raise ValueError(
                "near_minus_near_mul_far requires fixed_residual_alpha."
            )
        return near - fixed_residual_alpha * (near * far)
    if different_view == "near_minus_projected_far":
        if fixed_residual_alpha is None:
            raise ValueError(
                "near_minus_projected_far requires fixed_residual_alpha."
            )
        return near - fixed_residual_alpha * _project_near_onto_far(near, far)
    if different_view == "near_minus_normalized_far":
        if fixed_residual_alpha is None:
            raise ValueError(
                "near_minus_normalized_far requires fixed_residual_alpha."
            )
        return near - fixed_residual_alpha * _safe_l2_normalize(far)
    if different_view == "joint_pair_norm_residual":
        if fixed_residual_alpha is None:
            raise ValueError(
                "joint_pair_norm_residual requires fixed_residual_alpha."
            )
        residual = near - fixed_residual_alpha * far
        return residual / _joint_pair_rms_scale(near, far)
    if different_view == "joint_pair_sum_norm_residual":
        if fixed_residual_alpha is None:
            raise ValueError(
                "joint_pair_sum_norm_residual requires fixed_residual_alpha."
            )
        residual = near - fixed_residual_alpha * far
        return residual / _joint_pair_sum_scale(near, far)
    if different_view == "near_minus_0.5_far":
        return near - 0.5 * far
    if different_view == "near_with_far":
        return torch.cat([near, far], dim=-1)
    if different_view == "near_mix_far":
        return 0.5 * near + 0.5 * far
    if different_view == "near_div_far":
        eps = torch.finfo(far.dtype).eps if far.is_floating_point() else 1e-12
        safe_far = torch.where(
            far.abs() < eps,
            torch.where(far < 0, -torch.ones_like(far), torch.ones_like(far))
            * eps,
            far,
        )
        return near / safe_far
    if different_view == "near_dot_far":
        return (near * far).sum(dim=-1, keepdim=True)
    if different_view == "near_far_qk_scaled_dot":
        dim = near.shape[-1]
        query = near + far
        key = far + near
        return (query * key).sum(dim=-1, keepdim=True) / (dim ** 0.5)
    if different_view == "near_mul_far":
        return near * far
    if different_view == "near_plus_far":
        return near + far
    if different_view == "near_far_concat_cross_mul":
        query = torch.cat([near, far], dim=-1)
        key = torch.cat([far, near], dim=-1)
        return query * key
    if different_view == "near_far_concat_cross_mul_rp":
        query = torch.cat([near, far], dim=-1)
        key = torch.cat([far, near], dim=-1)
        return deterministic_random_projection(
            query * key,
            output_dim=near.shape[-1],
        )

    raise AssertionError("unreachable")


def collapse_paired_sequence(values: Sequence[T]) -> list[T]:
    """Keep one metadata entry per near/far pair."""
    if len(values) % 2 != 0:
        raise ValueError(
            f"Paired metadata length must be even, got {len(values)}."
        )
    return list(values[0::2])
