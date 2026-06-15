from __future__ import annotations

import argparse
import csv
import pickle
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.prepare_dcase2026 import get_dcase2026  # noqa: E402
from src.residual_view.differet_view import collapse_paired_sequence  # noqa: E402


ALPHA = 0.5
MEMMIX_ALPHA = 0.9
N_MIX_SUPPORT = 990
PRPS_K = 128
DECISION_Q = 0.95
PROJECTION_SEED = 20260614
EVAL_MACHINES = [
    "BlowerDustCollector",
    "Sander",
    "SewingMachine",
    "ToothBrush",
    "ToyDrone",
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class EncoderSpec:
    slug: str
    display: str
    cache_root: str
    layer: int


@dataclass(frozen=True)
class SystemSpec:
    label: str
    encoders: tuple[str, ...]
    use_prps: bool


ENCODERS = {
    "beats_iter3_l6": EncoderSpec(
        slug="beats_iter3_l6",
        display="BEATs iter3 L6",
        cache_root="cache_memory/train_features_beats_iter3_inputnear_differentview_pair",
        layer=6,
    ),
    "beats_iter3_l12": EncoderSpec(
        slug="beats_iter3_l12",
        display="BEATs iter3 L12",
        cache_root="cache_memory/train_features_beats_iter3_inputnear_differentview_pair",
        layer=12,
    ),
    "dasheng_base": EncoderSpec(
        slug="dasheng_base",
        display="DaSheng-base",
        cache_root="cache_memory/train_features_dasheng_base_inputnear_differentview_pair",
        layer=1,
    ),
    "sslam_l6": EncoderSpec(
        slug="sslam_l6",
        display="SSLAM L6",
        cache_root="cache_memory/train_features_sslam_inputnear_differentview_pair",
        layer=6,
    ),
    "sslam_l12": EncoderSpec(
        slug="sslam_l12",
        display="SSLAM L12",
        cache_root="cache_memory/train_features_sslam_inputnear_differentview_pair",
        layer=12,
    ),
}


SYSTEMS = [
    SystemSpec(
        label="Kim_LUDO_task2_1",
        encoders=("sslam_l12",),
        use_prps=True,
    ),
    SystemSpec(
        label="Kim_LUDO_task2_2",
        encoders=(
            "beats_iter3_l6",
            "beats_iter3_l12",
            "dasheng_base",
            "sslam_l6",
            "sslam_l12",
        ),
        use_prps=True,
    ),
]


def log(message: str) -> None:
    print(message, flush=True)


def load_pair_features(spec: EncoderSpec, split: str, machine: str) -> torch.Tensor:
    prefix = "train" if split in {"train", "eval_train"} else split
    path = (
        ROOT
        / spec.cache_root
        / f"temp_dcase2026_{split}"
        / f"{prefix}_temporal_{machine}.pkl"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Missing pair cache: {path}\n"
            "Generate it first with run_residual_view.py using "
            "`--different_view fixed_residual_view --fixed_residual_alpha 0.5`."
        )
    with path.open("rb") as handle:
        return pickle.load(handle).float()


def make_views(pair_features: torch.Tensor, layer: int) -> dict[str, torch.Tensor]:
    layer_features = pair_features[layer - 1]
    near = layer_features[0::2].flatten(1).contiguous()
    far = layer_features[1::2].flatten(1).contiguous()
    residual = near - ALPHA * far
    scale = (near * far).sum(dim=1, keepdim=True) / (
        (far * far).sum(dim=1, keepdim=True) + 1e-12
    )
    key = near - scale * far
    return {"near": near, "far": far, "residual": residual, "key": key}


def dist_matrix(x: torch.Tensor, y: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    x = x.to(DEVICE, non_blocking=True).float()
    y = y.to(DEVICE, non_blocking=True).float()
    y_t = y.T
    y_norm = (y**2).sum(dim=1, keepdim=True).T
    rows = []
    for start in range(0, x.shape[0], chunk):
        x_chunk = x[start : start + chunk]
        x_norm = (x_chunk**2).sum(dim=1, keepdim=True)
        dist_sq = x_norm + y_norm - 2 * x_chunk @ y_t
        rows.append(torch.sqrt(torch.clamp(dist_sq, min=0.0)).cpu())
    return torch.cat(rows, dim=0)


def nearest_scores(test_features: torch.Tensor, bank: torch.Tensor) -> np.ndarray:
    dist = dist_matrix(test_features, bank)
    return torch.topk(dist, k=1, dim=1, largest=False).values.numpy().reshape(-1)


def project(features: torch.Tensor, output_dim: int = 256) -> torch.Tensor:
    features = features.to(DEVICE, non_blocking=True).float()
    generator = torch.Generator(device=DEVICE).manual_seed(PROJECTION_SEED)
    projection = torch.randn(
        features.shape[1],
        output_dim,
        generator=generator,
        device=DEVICE,
        dtype=torch.float32,
    )
    projection = projection / (features.shape[1] ** 0.5)
    return (features @ projection).cpu()


def kcenter_order(features: torch.Tensor, k: int) -> torch.Tensor:
    features = features.to(DEVICE, non_blocking=True).float()
    k = min(k, features.shape[0])
    center = features.mean(dim=0, keepdim=True)
    first = torch.argmin(((features - center) ** 2).sum(dim=1)).item()
    selected = [first]
    min_dist = ((features - features[first]) ** 2).sum(dim=1)
    for _ in range(1, k):
        idx = torch.argmax(min_dist).item()
        selected.append(idx)
        dist = ((features - features[idx]) ** 2).sum(dim=1)
        min_dist = torch.minimum(min_dist, dist)
    return torch.tensor(selected, dtype=torch.long)


def make_memmix(train_features: torch.Tensor, train_domains: np.ndarray) -> torch.Tensor:
    source_idx = torch.from_numpy(np.flatnonzero(train_domains == "source")).long()
    target_idx = torch.from_numpy(np.flatnonzero(train_domains == "target")).long()
    source = train_features[source_idx]
    target = train_features[target_idx]
    k = min(N_MIX_SUPPORT, source.shape[0])
    dist = dist_matrix(target, source)
    _, topk = torch.topk(dist, k=k, dim=1, largest=False)
    mixed = []
    for target_i, source_indices in enumerate(topk):
        nearest = source[source_indices]
        mixed.append(MEMMIX_ALPHA * target[target_i] + (1.0 - MEMMIX_ALPHA) * nearest)
    return torch.cat(mixed, dim=0)


def loo_scores(
    train_features: torch.Tensor,
    bank_features: torch.Tensor,
    bank_indices: torch.Tensor | None = None,
) -> np.ndarray:
    dist = dist_matrix(train_features, bank_features)
    if bank_indices is None:
        dist[:, : train_features.shape[0]].fill_diagonal_(float("inf"))
    else:
        index_to_col = {int(idx): col for col, idx in enumerate(bank_indices.tolist())}
        rows = []
        cols = []
        for row_idx in range(train_features.shape[0]):
            col = index_to_col.get(row_idx)
            if col is not None:
                rows.append(row_idx)
                cols.append(col)
        if rows:
            dist[
                torch.tensor(rows, dtype=torch.long),
                torch.tensor(cols, dtype=torch.long),
            ] = float("inf")
    return torch.topk(dist, k=1, dim=1, largest=False).values.numpy().reshape(-1)


def machine_meta(machine: str) -> dict[str, object]:
    train_data = get_dcase2026("eval_train")["train"]
    test_data = get_dcase2026("train")["eval"]
    train_mask = np.asarray(train_data["machine_names"]) == machine
    test_mask = np.asarray(test_data["machine_names"]) == machine
    return {
        "train_domains": np.asarray(
            collapse_paired_sequence(np.asarray(train_data["source_list"])[train_mask])
        ),
        "test_files": [
            Path(path).name for path in np.asarray(test_data["file_list"])[test_mask]
        ],
    }


def residual_scores_for_encoder(
    spec: EncoderSpec,
    machine: str,
) -> tuple[np.ndarray, np.ndarray]:
    meta = machine_meta(machine)
    train_views = make_views(load_pair_features(spec, "eval_train", machine), spec.layer)
    test_views = make_views(load_pair_features(spec, "eval", machine), spec.layer)
    train_residual = train_views["residual"]
    test_residual = test_views["residual"]
    synthetic = make_memmix(train_residual, np.asarray(meta["train_domains"]))
    full_bank = torch.cat([train_residual, synthetic], dim=0)
    return (
        nearest_scores(test_residual, full_bank),
        loo_scores(train_residual, full_bank),
    )


def prps_scores_for_encoder(
    spec: EncoderSpec,
    machine: str,
) -> tuple[np.ndarray, np.ndarray]:
    meta = machine_meta(machine)
    train_views = make_views(load_pair_features(spec, "eval_train", machine), spec.layer)
    test_views = make_views(load_pair_features(spec, "eval", machine), spec.layer)
    train_residual = train_views["residual"]
    test_residual = test_views["residual"]
    train_key = train_views["key"]
    synthetic = make_memmix(train_residual, np.asarray(meta["train_domains"]))
    full_bank = torch.cat([train_residual, synthetic], dim=0)
    test_full = nearest_scores(test_residual, full_bank)
    train_full = loo_scores(train_residual, full_bank)

    indices = kcenter_order(project(train_key), PRPS_K)
    core_bank = train_residual[indices]
    test_core = nearest_scores(test_residual, core_bank)
    train_core = loo_scores(train_residual, core_bank, indices)
    return 0.5 * test_full + 0.5 * test_core, 0.5 * train_full + 0.5 * train_core


def system_scores(system: SystemSpec, machine: str) -> tuple[np.ndarray, np.ndarray]:
    test_scores = []
    train_scores = []
    for encoder_key in system.encoders:
        spec = ENCODERS[encoder_key]
        log(
            f"[score] system={system.label} encoder={spec.display} "
            f"prps={system.use_prps} machine={machine}"
        )
        if system.use_prps:
            test, train = prps_scores_for_encoder(spec, machine)
        else:
            test, train = residual_scores_for_encoder(spec, machine)
        test_scores.append(test)
        train_scores.append(train)
    return np.mean(np.stack(test_scores), axis=0), np.mean(np.stack(train_scores), axis=0)


def write_score_csv(path: Path, files: list[str], values: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for filename, value in zip(files, values):
            writer.writerow([filename, f"{float(value):.8f}"])


def write_decision_csv(path: Path, files: list[str], values: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for filename, value in zip(files, values):
            writer.writerow([filename, str(int(value))])


def copy_meta(system: SystemSpec, output_dir: Path) -> None:
    source = ROOT / "submissions" / "task2" / system.label / f"{system.label}.meta.yaml"
    target = output_dir / system.label / f"{system.label}.meta.yaml"
    if source.exists():
        shutil.copy2(source, target)


def build(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"[device] {DEVICE}")
    for system in SYSTEMS:
        system_dir = output_dir / system.label
        system_dir.mkdir(parents=True, exist_ok=True)
        for machine in EVAL_MACHINES:
            meta = machine_meta(machine)
            scores, train_scores = system_scores(system, machine)
            threshold = float(np.quantile(train_scores, DECISION_Q))
            decisions = (scores > threshold).astype(int)
            files = list(meta["test_files"])
            write_score_csv(
                system_dir / f"anomaly_score_{machine}_section_00_test.csv",
                files,
                scores,
            )
            write_decision_csv(
                system_dir / f"decision_result_{machine}_section_00_test.csv",
                files,
                decisions,
            )
            log(
                f"[decision] {system.label} {machine}: "
                f"q={DECISION_Q} threshold={threshold:.6f} ones={int(decisions.sum())}"
            )
        copy_meta(system, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final DCASE2026 Task 2 Residual View/PRPS package from pair caches."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "out" / "task2_reproduced",
        help="Output directory for reproduced DCASE-format system folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build(args.output_dir)


if __name__ == "__main__":
    main()
