from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

DEFAULT_INSTRUCTION = "Transfer the cube to the target position."


def log_step(message: str) -> None:
    print(f"[coreset] {message}", flush=True)


@dataclass(frozen=True)
class CoresetConfig:
    dataset_dir: Path
    output_dir: Path
    baseline_output_dir: Path | None = None
    camera_key: str = "observation.images.top"
    instruction: str = DEFAULT_INSTRUCTION
    arm: str = "right"
    coreset_fraction: float = 0.10
    seed: int = 42
    batch_size: int = 64
    epochs: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.20
    pretrained: bool = True
    language_dim: int = 128
    hidden_dim: int = 256
    num_workers: int = 0
    device: str = "auto"
    temporal_window: int = 3
    action_weight: float = 0.35
    state_weight: float = 0.25
    vision_weight: float = 0.25
    rarity_weight: float = 0.15
    rarity_neighbors: int = 10


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - minimum) / (maximum - minimum)).astype(np.float32)


def select_action_arm(actions: np.ndarray, arm: str = "right") -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"Expected action shape [N, 14], got {actions.shape}.")
    if arm == "left":
        return actions[:, :7]
    if arm == "right":
        return actions[:, 7:]
    raise ValueError("arm must be either 'left' or 'right'.")


def build_language_features(text: str, rows: int, dim: int = 128) -> np.ndarray:
    if rows < 0:
        raise ValueError("rows must be non-negative.")
    if dim <= 0:
        raise ValueError("dim must be positive.")

    vector = np.zeros(dim, dtype=np.float32)
    for token in tokenize_instruction(text):
        bucket = stable_hash(token) % dim
        sign = 1.0 if (stable_hash(token + "::sign") % 2 == 0) else -1.0
        vector[bucket] += sign

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return np.repeat(vector[None, :], rows, axis=0)


def tokenize_instruction(text: str) -> list[str]:
    normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [part for part in normalized.split() if part]


def stable_hash(text: str) -> int:
    value = 2166136261
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def compute_predictive_diversity_scores(
    vision_features: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    action_weight: float = 0.35,
    state_weight: float = 0.25,
    vision_weight: float = 0.25,
    rarity_weight: float = 0.15,
    rarity_neighbors: int = 10,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not (len(vision_features) == len(states) == len(actions)):
        raise ValueError("vision_features, states, and actions must have the same length.")
    if len(actions) == 0:
        raise ValueError("Cannot score an empty dataset.")

    action_delta = pairwise_step_l2(actions)
    state_delta = pairwise_step_l2(states)
    vision_delta = pairwise_step_cosine_distance(vision_features)
    rarity = approximate_feature_rarity(vision_features, neighbors=rarity_neighbors)

    parts = {
        "action_delta": minmax_normalize(action_delta),
        "state_delta": minmax_normalize(state_delta),
        "vision_delta": minmax_normalize(vision_delta),
        "rarity": minmax_normalize(rarity),
    }
    score = (
        action_weight * parts["action_delta"]
        + state_weight * parts["state_delta"]
        + vision_weight * parts["vision_delta"]
        + rarity_weight * parts["rarity"]
    ).astype(np.float32)
    return score, parts


def pairwise_step_l2(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    deltas = np.zeros(len(values), dtype=np.float32)
    if len(values) > 1:
        deltas[1:] = np.linalg.norm(values[1:] - values[:-1], axis=1)
    return deltas


def pairwise_step_cosine_distance(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    distances = np.zeros(len(values), dtype=np.float32)
    if len(values) <= 1:
        return distances
    norms = np.linalg.norm(values, axis=1)
    current_valid = norms[1:] > 1e-12
    previous_valid = norms[:-1] > 1e-12
    valid = current_valid & previous_valid
    one_sided_valid = current_valid ^ previous_valid
    distances[1:][one_sided_valid] = 1.0
    if not np.any(valid):
        return distances.astype(np.float32)
    normalized = values / np.maximum(norms[:, None], 1e-12)
    similarity = np.sum(normalized[1:] * normalized[:-1], axis=1)
    distances[1:][valid] = 1.0 - np.clip(similarity[valid], -1.0, 1.0)
    return distances.astype(np.float32)


def approximate_feature_rarity(values: np.ndarray, neighbors: int = 10) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) <= 1:
        return np.zeros(len(values), dtype=np.float32)

    normalized = l2_normalize(values)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)
    k = max(1, min(neighbors, len(values) - 1))
    nearest = np.partition(similarity, -k, axis=1)[:, -k:]
    density = nearest.mean(axis=1)
    return (1.0 - density).astype(np.float32)


def l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def select_top_with_temporal_suppression(
    scores: np.ndarray, count: int, window: int = 3
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if count <= 0:
        return np.array([], dtype=np.int64)
    count = min(count, len(scores))
    window = max(0, int(window))

    available = np.ones(len(scores), dtype=bool)
    selected: list[int] = []
    order = np.argsort(-scores, kind="mergesort")
    for index in order:
        index = int(index)
        if not available[index]:
            continue
        selected.append(index)
        start = max(0, index - window)
        end = min(len(scores), index + window + 1)
        available[start:end] = False
        if len(selected) == count:
            break

    if len(selected) < count:
        chosen = set(selected)
        for index in order:
            index = int(index)
            if index not in chosen:
                selected.append(index)
                if len(selected) == count:
                    break
    return np.array(sorted(selected), dtype=np.int64)


def run_coreset(config: CoresetConfig) -> dict:
    log_step("importing training dependencies")
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm

    log_step(f"using device setting: {config.device}")
    set_seed(config.seed)
    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    log_step(f"loading dataset info from {config.dataset_dir}")
    info = load_dataset_info(config.dataset_dir)
    log_step("dataset info loaded")
    log_step(f"loading parquet metadata from {config.dataset_dir / 'data'}")
    metadata = load_metadata(config.dataset_dir).sort_values("index").reset_index(drop=True)
    log_step(
        f"loaded {len(metadata)} frames from {metadata['episode_index'].nunique()} episodes"
    )
    actions_14d = np.stack(metadata["action"].to_numpy()).astype(np.float32)
    states = np.stack(metadata["observation.state"].to_numpy()).astype(np.float32)
    targets = select_action_arm(actions_14d, config.arm)

    feature_path = config.output_dir / "features_vision.npy"
    target_path = config.output_dir / "targets_action.npy"
    full_rows_path = config.output_dir / "full_rows.csv"
    reusable_feature_path = (
        config.baseline_output_dir / "features_vision.npy"
        if config.baseline_output_dir
        else None
    )

    if feature_path.exists() and target_path.exists():
        log_step(f"loading cached vision features from {feature_path}")
        vision_features = np.load(feature_path)
    elif reusable_feature_path and reusable_feature_path.exists():
        log_step(f"loading reusable full-dataset features from {reusable_feature_path}")
        vision_features = np.load(reusable_feature_path)
        if len(vision_features) != len(metadata):
            raise ValueError(
                f"Reusable feature file has {len(vision_features)} rows, expected {len(metadata)} full-dataset rows."
            )
        np.save(feature_path, vision_features)
        np.save(target_path, targets)
        metadata.to_csv(full_rows_path, index=False)
    else:
        log_step("building frozen ResNet-18 encoder")
        encoder, transform = build_frozen_resnet18(config.pretrained, device)
        log_step("extracting full-dataset vision features; this is the slowest step")
        vision_features = extract_video_features(
            dataset_dir=config.dataset_dir,
            info=info,
            rows=metadata,
            camera_key=config.camera_key,
            encoder=encoder,
            transform=transform,
            device=device,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            progress=tqdm,
        )
        np.save(feature_path, vision_features)
        np.save(target_path, targets)
        metadata.to_csv(full_rows_path, index=False)
        log_step(f"saved vision features to {feature_path}")

    all_episode_indices = metadata["episode_index"].to_numpy()
    candidate_idx, val_idx = make_train_val_split(
        all_episode_indices, config.val_fraction, config.seed
    )
    coreset_count = max(1, math.ceil(len(metadata) * config.coreset_fraction))
    log_step(
        f"selecting {coreset_count} coreset frames from {len(candidate_idx)} candidate frames; validation frames={len(val_idx)}"
    )

    candidate_scores, candidate_parts = compute_predictive_diversity_scores(
        vision_features[candidate_idx],
        states[candidate_idx],
        targets[candidate_idx],
        action_weight=config.action_weight,
        state_weight=config.state_weight,
        vision_weight=config.vision_weight,
        rarity_weight=config.rarity_weight,
        rarity_neighbors=config.rarity_neighbors,
    )
    local_selected = select_top_with_temporal_suppression(
        candidate_scores, count=coreset_count, window=config.temporal_window
    )
    train_idx = np.sort(candidate_idx[local_selected])

    selected_scores = candidate_scores[local_selected]
    selected_parts = {name: values[local_selected] for name, values in candidate_parts.items()}
    write_coreset_index_file(
        path=config.output_dir / "coreset_indices.csv",
        metadata=metadata,
        train_idx=train_idx,
        scores=selected_scores,
        parts=selected_parts,
    )
    log_step(f"saved selected frame list to {config.output_dir / 'coreset_indices.csv'}")

    language_features = build_language_features(
        config.instruction, rows=len(metadata), dim=config.language_dim
    )
    features = np.concatenate([vision_features, language_features], axis=1).astype(
        np.float32
    )

    log_step("training MLP on selected coreset")
    metrics, model = train_mlp(
        features=features,
        targets=targets,
        train_idx=train_idx,
        val_idx=val_idx,
        hidden_dim=config.hidden_dim,
        batch_size=config.batch_size,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=device,
        torch=torch,
        nn=nn,
        DataLoader=DataLoader,
        TensorDataset=TensorDataset,
    )
    torch.save(model.state_dict(), config.output_dir / "mlp_action_regressor.pt")
    log_step(f"saved trained MLP to {config.output_dir / 'mlp_action_regressor.pt'}")

    result = {
        "config": {
            **asdict(config),
            "dataset_dir": str(config.dataset_dir),
            "output_dir": str(config.output_dir),
            "baseline_output_dir": str(config.baseline_output_dir)
            if config.baseline_output_dir
            else None,
        },
        "dataset": {
            "total_episodes": int(metadata["episode_index"].nunique()),
            "total_frames": int(len(metadata)),
            "candidate_frame_count": int(len(candidate_idx)),
            "validation_frame_count": int(len(val_idx)),
            "coreset_frame_count": int(len(train_idx)),
            "coreset_fraction_of_full_data": float(len(train_idx) / len(metadata)),
            "target_arm": config.arm,
            "target_action_dim": 7,
        },
        "coreset": {
            "method": "PD-Coreset",
            "description": "Brain-inspired predictive-diversity coreset using action/state/vision change, feature rarity, and temporal suppression.",
            "weights": {
                "action_delta": config.action_weight,
                "state_delta": config.state_weight,
                "vision_delta": config.vision_weight,
                "rarity": config.rarity_weight,
            },
            "temporal_window": config.temporal_window,
            "rarity_neighbors": config.rarity_neighbors,
            "mean_selected_score": float(np.mean(selected_scores)),
        },
        "metrics": metrics,
    }
    with (config.output_dir / "coreset_results.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log_step(f"saved results to {config.output_dir / 'coreset_results.json'}")
    return result


def write_coreset_index_file(
    path: Path,
    metadata,
    train_idx: np.ndarray,
    scores: np.ndarray,
    parts: dict[str, np.ndarray],
) -> None:
    import pandas as pd

    frame = metadata.iloc[train_idx][
        ["index", "episode_index", "frame_index", "timestamp", "task_index"]
    ].copy()
    frame.insert(0, "selection_rank", np.arange(1, len(frame) + 1))
    frame["coreset_score"] = scores
    for name, values in parts.items():
        frame[name] = values
    frame.to_csv(path, index=False)


def load_dataset_info(dataset_dir: Path) -> dict:
    with (dataset_dir / "meta" / "info.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata(dataset_dir: Path):
    log_step("importing pandas and pyarrow")
    import pandas as pd
    import pyarrow.parquet as pq

    parquet_paths = sorted((dataset_dir / "data").glob("*/*.parquet"))
    log_step(f"found {len(parquet_paths)} parquet files")
    frames = []
    for parquet_path in parquet_paths:
        log_step(f"reading parquet file: {parquet_path}")
        started_at = time.perf_counter()
        parquet_file = pq.ParquetFile(parquet_path)
        table = parquet_file.read(
            columns=[
                "observation.state",
                "action",
                "episode_index",
                "frame_index",
                "timestamp",
                "next.done",
                "index",
                "task_index",
            ],
            use_threads=False,
        )
        frame = table.to_pandas()
        frame["data_file"] = str(parquet_path)
        frames.append(frame)
        elapsed = time.perf_counter() - started_at
        log_step(f"loaded {len(frame)} rows from {parquet_path.name} in {elapsed:.2f}s")
    if not frames:
        raise FileNotFoundError(f"No parquet files found under {dataset_dir / 'data'}.")
    metadata = pd.concat(frames, ignore_index=True)
    log_step(f"concatenated metadata rows: {len(metadata)}")
    return metadata


def make_train_val_split(
    episode_indices: np.ndarray, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    episodes = np.unique(episode_indices.astype(np.int64))
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1).")
    val_count = max(1, math.ceil(len(episodes) * val_fraction))
    rng = random.Random(seed + 1009)
    val_episodes = sorted(rng.sample([int(ep) for ep in episodes], val_count))
    val_mask = np.isin(episode_indices, val_episodes)
    train_idx = np.flatnonzero(~val_mask)
    val_idx = np.flatnonzero(val_mask)
    if len(train_idx) == 0 or len(val_idx) == 0:
        indices = np.arange(len(episode_indices))
        rng_np = np.random.default_rng(seed)
        rng_np.shuffle(indices)
        fallback_val_count = max(1, int(math.ceil(len(indices) * val_fraction)))
        val_idx = np.sort(indices[:fallback_val_count])
        train_idx = np.sort(indices[fallback_val_count:])
    return train_idx, val_idx


def build_frozen_resnet18(pretrained: bool, device):
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    transform = weights.transforms() if weights else ResNet18_Weights.DEFAULT.transforms()
    return model, transform


def assert_pyav_available() -> None:
    try:
        importlib.import_module("av")
    except ImportError as exc:
        raise RuntimeError(
            "PyAV is required to decode the ALOHA mp4 videos. Install it in the "
            "active environment with: pip install av"
        ) from exc


def extract_video_features(
    dataset_dir: Path,
    info: dict,
    rows,
    camera_key: str,
    encoder,
    transform,
    device,
    batch_size: int,
    num_workers: int,
    progress,
) -> np.ndarray:
    assert_pyav_available()
    video_path = resolve_video_path(dataset_dir, info, camera_key)
    frame_indices = rows["index"].to_numpy(dtype=np.int64)
    log_step(f"reading video frames from {video_path}")
    return extract_video_features_streaming(
        video_path=video_path,
        frame_indices=frame_indices,
        encoder=encoder,
        transform=transform,
        device=device,
        batch_size=batch_size,
        progress=progress,
    )


def resolve_video_path(dataset_dir: Path, info: dict, camera_key: str) -> Path:
    pattern = info.get(
        "video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )
    relative = pattern.format(video_key=camera_key, chunk_index=0, file_index=0)
    path = dataset_dir / relative
    if path.exists():
        return path
    matches = sorted((dataset_dir / "videos" / camera_key).glob("*/*.mp4"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find video for camera {camera_key}: {path}")


def extract_video_features_streaming(
    video_path: Path,
    frame_indices: np.ndarray,
    encoder,
    transform,
    device,
    batch_size: int,
    progress,
) -> np.ndarray:
    import torch
    from torchvision.io import VideoReader

    target_positions: dict[int, list[int]] = {}
    for output_pos, frame_index in enumerate(frame_indices.tolist()):
        target_positions.setdefault(int(frame_index), []).append(output_pos)

    output_features: list[np.ndarray | None] = [None] * len(frame_indices)
    batch_frames = []
    batch_positions = []
    max_frame_index = int(frame_indices.max())

    try:
        reader = VideoReader(str(video_path), "video")
        iterator = enumerate(reader)
    except Exception as exc:
        print(f"VideoReader unavailable ({exc}); falling back to read_video.")
        return extract_video_features_in_memory(
            video_path, frame_indices, encoder, transform, device, batch_size, progress
        )

    for frame_number, frame in progress(
        iterator,
        desc="Scanning video frames",
        total=max_frame_index + 1,
    ):
        if frame_number > max_frame_index:
            break
        if frame_number not in target_positions:
            continue

        image = frame["data"]
        for output_pos in target_positions[frame_number]:
            batch_frames.append(image)
            batch_positions.append(output_pos)
        if len(batch_frames) >= batch_size:
            encoded = encode_frame_batch(batch_frames, encoder, transform, device, torch)
            for output_pos, feature in zip(batch_positions, encoded):
                output_features[output_pos] = feature
            batch_frames = []
            batch_positions = []

    if batch_frames:
        encoded = encode_frame_batch(batch_frames, encoder, transform, device, torch)
        for output_pos, feature in zip(batch_positions, encoded):
            output_features[output_pos] = feature

    missing = [i for i, feature in enumerate(output_features) if feature is None]
    if missing:
        raise ValueError(
            f"Could not decode {len(missing)} requested frames from {video_path}; first missing output index is {missing[0]}."
        )
    return np.stack(output_features).astype(np.float32)


def extract_video_features_in_memory(
    video_path: Path,
    frame_indices: np.ndarray,
    encoder,
    transform,
    device,
    batch_size: int,
    progress,
) -> np.ndarray:
    import torch

    video, _, _ = read_video_compat(video_path)
    if len(video) == 0:
        raise ValueError(f"No frames decoded from {video_path}.")

    features = []
    for start in progress(
        range(0, len(frame_indices), batch_size),
        desc="Extracting ResNet-18 features",
    ):
        batch_indices = frame_indices[start : start + batch_size]
        batch = video[batch_indices].permute(0, 3, 1, 2)
        encoded = encode_frame_batch(batch, encoder, transform, device, torch)
        features.append(encoded)
    return np.concatenate(features, axis=0).astype(np.float32)


def read_video_compat(video_path: Path):
    from torchvision.io import read_video

    return read_video(str(video_path), pts_unit="sec", output_format="THWC")


def encode_frame_batch(frames, encoder, transform, device, torch) -> np.ndarray:
    if isinstance(frames, list):
        batch = torch.stack(frames, dim=0)
    else:
        batch = frames
    batch = batch.float() / 255.0
    if batch.ndim != 4:
        raise ValueError(f"Expected frame batch with 4 dimensions, got {batch.shape}.")
    if batch.shape[-1] == 3:
        batch = batch.permute(0, 3, 1, 2)
    with torch.inference_mode():
        batch = transform(batch).to(device)
        return encoder(batch).cpu().numpy().astype(np.float32)


def train_mlp(
    features: np.ndarray,
    targets: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    hidden_dim: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device,
    torch,
    nn,
    DataLoader,
    TensorDataset,
):
    x_train = torch.from_numpy(features[train_idx]).float()
    y_train = torch.from_numpy(targets[train_idx]).float()
    x_val = torch.from_numpy(features[val_idx]).float().to(device)
    y_val = torch.from_numpy(targets[val_idx]).float().to(device)

    model = nn.Sequential(
        nn.Linear(features.shape[1], hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.10),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, targets.shape[1]),
    ).to(device)

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()
    history = []
    best_val_mse = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch_x)
            seen += len(batch_x)

        model.eval()
        with torch.inference_mode():
            val_prediction = model(x_val)
            val_mse = criterion(val_prediction, y_val).item()
        train_mse = running_loss / max(1, seen)
        best_val_mse = min(best_val_mse, val_mse)
        history.append(
            {"epoch": epoch, "train_mse": float(train_mse), "val_mse": float(val_mse)}
        )
        print(f"epoch={epoch:03d} train_mse={train_mse:.6f} val_mse={val_mse:.6f}")

    metrics = {
        "final_train_mse": history[-1]["train_mse"],
        "final_val_mse": history[-1]["val_mse"],
        "best_val_mse": float(best_val_mse),
        "history": history,
    }
    return metrics, model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def parse_args() -> CoresetConfig:
    parser = argparse.ArgumentParser(
        description="PD-Coreset: select an informative 10% ALOHA coreset and retrain the same MLP action regressor."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("aloha_sim_transfer_cube_human"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/coreset_pd10_resnet18"))
    parser.add_argument("--baseline-output-dir", type=Path, default=None)
    parser.add_argument("--camera-key", default="observation.images.top")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--arm", choices=["left", "right"], default="right")
    parser.add_argument("--coreset-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--language-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temporal-window", type=int, default=3)
    parser.add_argument("--action-weight", type=float, default=0.35)
    parser.add_argument("--state-weight", type=float, default=0.25)
    parser.add_argument("--vision-weight", type=float, default=0.25)
    parser.add_argument("--rarity-weight", type=float, default=0.15)
    parser.add_argument("--rarity-neighbors", type=int, default=10)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    return CoresetConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        baseline_output_dir=args.baseline_output_dir,
        camera_key=args.camera_key,
        instruction=args.instruction,
        arm=args.arm,
        coreset_fraction=args.coreset_fraction,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        pretrained=not args.no_pretrained,
        language_dim=args.language_dim,
        hidden_dim=args.hidden_dim,
        num_workers=args.num_workers,
        device=args.device,
        temporal_window=args.temporal_window,
        action_weight=args.action_weight,
        state_weight=args.state_weight,
        vision_weight=args.vision_weight,
        rarity_weight=args.rarity_weight,
        rarity_neighbors=args.rarity_neighbors,
    )


def main() -> None:
    result = run_coreset(parse_args())
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
