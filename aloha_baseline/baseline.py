from __future__ import annotations

import argparse
import importlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_INSTRUCTION = "Transfer the cube to the target position."


@dataclass(frozen=True)
class BaselineConfig:
    dataset_dir: Path
    output_dir: Path
    camera_key: str = "observation.images.top"
    instruction: str = DEFAULT_INSTRUCTION
    arm: str = "right"
    sample_fraction: float = 0.10
    seed: int = 42
    batch_size: int = 64
    epochs: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.20
    model_name: str = "resnet18"
    pretrained: bool = True
    language_dim: int = 128
    hidden_dim: int = 256
    num_workers: int = 0
    device: str = "auto"


def select_episode_subset(
    episodes: Iterable[int], fraction: float = 0.10, seed: int = 42
) -> list[int]:
    """Randomly sample a fraction of episodes, keeping at least one episode."""
    unique_episodes = sorted({int(ep) for ep in episodes})
    if not unique_episodes:
        raise ValueError("No episodes found to sample.")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")

    count = max(1, math.ceil(len(unique_episodes) * fraction))
    rng = random.Random(seed)
    return sorted(rng.sample(unique_episodes, count))


def select_action_arm(actions: np.ndarray, arm: str = "right") -> np.ndarray:
    """Select a single 7-DoF arm action from ALOHA's 14-DoF bimanual action."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"Expected action shape [N, 14], got {actions.shape}.")
    if arm == "left":
        return actions[:, :7]
    if arm == "right":
        return actions[:, 7:]
    raise ValueError("arm must be either 'left' or 'right'.")


def build_language_features(text: str, rows: int, dim: int = 128) -> np.ndarray:
    """Create a deterministic lightweight instruction embedding.

    This baseline keeps the vision encoder pretrained/frozen and uses a simple
    hashed bag-of-words vector for the single task instruction. It avoids adding
    a heavy language model while still constructing [vision + language] inputs.
    """
    if rows < 0:
        raise ValueError("rows must be non-negative.")
    if dim <= 0:
        raise ValueError("dim must be positive.")

    vector = np.zeros(dim, dtype=np.float32)
    for token in _tokenize(text):
        bucket = _stable_hash(token) % dim
        sign = 1.0 if (_stable_hash(token + "::sign") % 2 == 0) else -1.0
        vector[bucket] += sign

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return np.repeat(vector[None, :], rows, axis=0)


def assert_pyav_available() -> None:
    """Fail early with a clear message when torchvision video decoding cannot run."""
    try:
        importlib.import_module("av")
    except ImportError as exc:
        raise RuntimeError(
            "PyAV is required to decode the ALOHA mp4 videos. Install it in the "
            "active conda environment with: conda install -c conda-forge av"
        ) from exc


def _tokenize(text: str) -> list[str]:
    normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [part for part in normalized.split() if part]


def _stable_hash(text: str) -> int:
    value = 2166136261
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def run_baseline(config: BaselineConfig) -> dict:
    import pandas as pd
    import pyarrow.parquet as pq
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision.io import read_video
    from torchvision.models import ResNet18_Weights, resnet18
    from tqdm import tqdm

    set_seed(config.seed)
    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    info = load_dataset_info(config.dataset_dir)
    metadata = load_metadata(config.dataset_dir)
    sampled_episodes = select_episode_subset(
        metadata["episode_index"].unique(), config.sample_fraction, config.seed
    )
    rows = metadata[metadata["episode_index"].isin(sampled_episodes)].copy()
    rows = rows.sort_values("index").reset_index(drop=True)

    actions = np.stack(rows["action"].to_numpy()).astype(np.float32)
    targets = select_action_arm(actions, config.arm)

    feature_path = config.output_dir / "features_vision.npy"
    target_path = config.output_dir / "targets_action.npy"
    meta_path = config.output_dir / "sampled_rows.csv"

    if feature_path.exists() and target_path.exists():
        vision_features = np.load(feature_path)
    else:
        encoder, transform = build_frozen_resnet18(config.pretrained, device)
        vision_features = extract_video_features(
            dataset_dir=config.dataset_dir,
            info=info,
            rows=rows,
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
        rows.to_csv(meta_path, index=False)

    language_features = build_language_features(
        config.instruction, rows=len(rows), dim=config.language_dim
    )
    features = np.concatenate([vision_features, language_features], axis=1).astype(
        np.float32
    )

    train_idx, val_idx = make_train_val_split(
        rows["episode_index"].to_numpy(), config.val_fraction, config.seed
    )
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
    result = {
        "config": {**asdict(config), "dataset_dir": str(config.dataset_dir), "output_dir": str(config.output_dir)},
        "dataset": {
            "total_episodes": int(metadata["episode_index"].nunique()),
            "sampled_episodes": [int(ep) for ep in sampled_episodes],
            "sampled_episode_count": len(sampled_episodes),
            "sampled_frame_count": int(len(rows)),
            "target_arm": config.arm,
            "target_action_dim": 7,
        },
        "metrics": metrics,
    }
    with (config.output_dir / "baseline_results.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def load_dataset_info(dataset_dir: Path) -> dict:
    with (dataset_dir / "meta" / "info.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata(dataset_dir: Path):
    import pandas as pd
    import pyarrow.parquet as pq

    frames = []
    for parquet_path in sorted((dataset_dir / "data").glob("*/*.parquet")):
        table = pq.read_table(parquet_path)
        frame = table.to_pandas()
        frame["data_file"] = str(parquet_path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No parquet files found under {dataset_dir / 'data'}.")
    return pd.concat(frames, ignore_index=True)


def build_frozen_resnet18(pretrained: bool, device):
    import torch
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
    import torch

    assert_pyav_available()
    video_path = resolve_video_path(dataset_dir, info, camera_key)
    frame_indices = rows["index"].to_numpy(dtype=np.int64)
    return extract_video_features_streaming(
        video_path=video_path,
        frame_indices=frame_indices,
        encoder=encoder,
        transform=transform,
        device=device,
        batch_size=batch_size,
        progress=progress,
    )


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

    for frame_number, frame in progress(iterator, desc="Scanning video frames"):
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


def resolve_video_path(dataset_dir: Path, info: dict, camera_key: str) -> Path:
    pattern = info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    relative = pattern.format(video_key=camera_key, chunk_index=0, file_index=0)
    path = dataset_dir / relative
    if not path.exists():
        matches = sorted((dataset_dir / "videos" / camera_key).glob("*/*.mp4"))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"Could not find video for camera {camera_key}: {path}")
    return path


def read_video_compat(video_path: Path):
    from torchvision.io import read_video

    return read_video(str(video_path), pts_unit="sec", output_format="THWC")


def make_train_val_split(
    episode_indices: np.ndarray, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    sampled_episodes = np.unique(episode_indices.astype(np.int64))
    val_episodes = select_episode_subset(sampled_episodes, val_fraction, seed + 1009)
    val_mask = np.isin(episode_indices, val_episodes)
    train_idx = np.flatnonzero(~val_mask)
    val_idx = np.flatnonzero(val_mask)
    if len(train_idx) == 0 or len(val_idx) == 0:
        indices = np.arange(len(episode_indices))
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        val_count = max(1, int(math.ceil(len(indices) * val_fraction)))
        val_idx = np.sort(indices[:val_count])
        train_idx = np.sort(indices[val_count:])
    return train_idx, val_idx


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


def parse_args() -> BaselineConfig:
    parser = argparse.ArgumentParser(
        description="Random 10% ALOHA baseline: frozen ResNet-18 features + instruction vector -> 7-DoF action MLP."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("aloha_sim_transfer_cube_human"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline_random10_resnet18"))
    parser.add_argument("--camera-key", default="observation.images.top")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--arm", choices=["left", "right"], default="right")
    parser.add_argument("--sample-fraction", type=float, default=0.10)
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
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    return BaselineConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        camera_key=args.camera_key,
        instruction=args.instruction,
        arm=args.arm,
        sample_fraction=args.sample_fraction,
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
    )


def main() -> None:
    result = run_baseline(parse_args())
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
