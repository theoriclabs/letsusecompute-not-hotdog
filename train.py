"""Train a tiny from-scratch CNN: hot dog vs not hot dog.

Dataset (preferred): ``theoriclabs/hot-dog-not-hot-dog`` on Hugging Face.
Fallback: binary cut of ``ethz/food101`` (Bossard et al., ECCV 2014).

Weights land under ``$COMPUTE_ARTIFACT_DIR`` with a ``.compute-artifact.json``
marker so ``compute artifacts get`` works after teardown. Optional Hub push
when ``push_to_hub=True`` and an HF write token is available.

    compute run train.py::train --gpu cheap --provider vastai --dry-run
    compute run train.py::train --gpu cheap --provider vastai --timeout 1200 --wait
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import compute

app = compute.App("not-hotdog")
image = compute.Image.cuda_pytorch().pip_install(
    "torchvision",
    "datasets",
    "huggingface_hub",
    "Pillow",
)

# Stored with `compute secrets set hf`. Injected only when a function lists
# `secrets=[hf_secret]`. Training the public dataset does not need it.
hf_secret = compute.Secret.from_name("hf")

DEFAULT_DATASET = "theoriclabs/hot-dog-not-hot-dog"
FALLBACK_DATASET = "ethz/food101"
WORKLOAD_SUBDIR = "not-hotdog"
ARTIFACT_NAME = "not-hotdog-cnn"
ARTIFACT_MARKER = ".compute-artifact.json"
DEFAULT_ARTIFACT_FALLBACK = Path("/tmp/compute-not-hotdog")
CLASS_NAMES = ("hot_dog", "not_hot_dog")


def _bridge_hf_token() -> None:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("hf")
    )
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)


def write_artifact_marker(
    directory: Path | str,
    *,
    name: str,
    kind: str,
    compatibility_key: str,
    metadata: dict[str, Any],
) -> Path:
    """Atomically write ``.compute-artifact.json`` (temp + flush + fsync + replace)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    marker_path = directory / ARTIFACT_MARKER
    payload = {
        "compatibility_key": compatibility_key,
        "kind": kind,
        "metadata": metadata,
        "name": name,
        "version": 1,
    }
    data = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=".compute-artifact.",
        suffix=".tmp",
        dir=str(directory),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, marker_path)
        dir_fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    for leftover in directory.glob(".compute-artifact.*.tmp"):
        try:
            leftover.unlink()
        except OSError:
            pass
    return marker_path


def resolve_artifact_dirs(env: Mapping[str, str] | None = None) -> Path:
    environ = os.environ if env is None else env
    base = environ.get("COMPUTE_ARTIFACT_DIR")
    root = Path(base) / WORKLOAD_SUBDIR if base else DEFAULT_ARTIFACT_FALLBACK
    root.mkdir(parents=True, exist_ok=True)
    return root


def _tiny_cnn(num_classes: int = 2):
    import torch.nn as nn

    # ~180k params. Random init — no pretrained weights.
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, num_classes),
    )


def _load_binary_dataset(dataset_id: str, max_train: int, max_test: int):
    from datasets import load_dataset
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        ds = load_dataset(dataset_id)
    except (RepositoryNotFoundError, FileNotFoundError, OSError) as err:
        # Only fall back when the preferred repo is missing — never on network
        # blips mid-download of a huge Food-101 pull.
        if dataset_id == FALLBACK_DATASET:
            raise
        print(f"dataset {dataset_id!r} unavailable ({err}); falling back to {FALLBACK_DATASET}", flush=True)
        food = load_dataset(FALLBACK_DATASET)
        hot_dog_id = food["train"].features["label"].str2int("hot_dog")

        def to_binary(example: dict[str, Any]) -> dict[str, Any]:
            return {
                "image": example["image"],
                "label": 0 if example["label"] == hot_dog_id else 1,
            }

        train_all = food["train"].map(to_binary, remove_columns=food["train"].column_names)
        test_all = food["validation"].map(
            to_binary, remove_columns=food["validation"].column_names
        )
        train_hot = [i for i, y in enumerate(train_all["label"]) if y == 0][: max_train // 2]
        train_not = [i for i, y in enumerate(train_all["label"]) if y == 1][: max_train // 2]
        test_hot = [i for i, y in enumerate(test_all["label"]) if y == 0][: max_test // 2]
        test_not = [i for i, y in enumerate(test_all["label"]) if y == 1][: max_test // 2]
        train = train_all.select(train_hot + train_not)
        test = test_all.select(test_hot + test_not)
        return train, test, FALLBACK_DATASET

    train = ds["train"]
    test = ds["test"] if "test" in ds else ds["validation"]
    return train.select(range(min(len(train), max_train))), test.select(
        range(min(len(test), max_test))
    ), dataset_id


def _train_impl(
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    image_size: int,
    max_train: int,
    max_test: int,
    dataset_id: str,
    seed: int,
    push_to_hub: bool,
    hub_repo: str,
) -> dict:
    """Train the Not Hotdog CNN and write weights as a compute artifact."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    _bridge_hf_token()
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required (torch.cuda.is_available() is False)")
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)

    train_ds, test_ds, resolved_dataset = _load_binary_dataset(dataset_id, max_train, max_test)

    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    class HFVision(Dataset):
        def __init__(self, split) -> None:
            self.split = split

        def __len__(self) -> int:
            return len(self.split)

        def __getitem__(self, idx: int):
            row = self.split[idx]
            img = row["image"].convert("RGB")
            x = tfm(img)
            y = int(row["label"])
            return x, y

    train_loader = DataLoader(HFVision(train_ds), batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(HFVision(test_ds), batch_size=batch_size, shuffle=False, num_workers=2)

    model = _tiny_cnn(num_classes=2).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach()) * xb.size(0)
            n += xb.size(0)
        train_loss = total_loss / max(n, 1)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += yb.size(0)
        acc = correct / max(total, 1)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "test_accuracy": acc})
        print(f"epoch {epoch + 1}/{epochs} loss={train_loss:.4f} acc={acc:.4f}", flush=True)

    out_dir = resolve_artifact_dirs()
    weights_path = out_dir / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": list(CLASS_NAMES),
            "image_size": image_size,
            "param_count": param_count,
            "dataset_id": resolved_dataset,
        },
        weights_path,
    )
    write_artifact_marker(
        out_dir,
        name=ARTIFACT_NAME,
        kind="model_weights",
        compatibility_key=f"not-hotdog-cnn-v1-{image_size}",
        metadata={
            "filename": weights_path.name,
            "param_count": param_count,
            "epochs": epochs,
            "test_accuracy": history[-1]["test_accuracy"] if history else None,
            "dataset_id": resolved_dataset,
            "class_names": list(CLASS_NAMES),
        },
    )

    hub_url = None
    if push_to_hub:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(hub_repo, exist_ok=True, private=False)
            api.upload_file(
                path_or_fileobj=str(weights_path),
                path_in_repo="model.pt",
                repo_id=hub_repo,
            )
            hub_url = f"https://huggingface.co/{hub_repo}"
        except Exception as err:  # noqa: BLE001 — publish is optional; keep weights via artifacts
            print(f"push_to_hub failed: {err}", flush=True)
            hub_url = None

    return {
        "ok": True,
        "compat": "not-hotdog-cnn",
        "device": str(device),
        "device_name": device_name,
        "dataset_id": resolved_dataset,
        "epochs": epochs,
        "param_count": param_count,
        "image_size": image_size,
        "train_size": len(train_ds),
        "test_size": len(test_ds),
        "history": history,
        "test_accuracy": history[-1]["test_accuracy"] if history else None,
        "artifact_dir": str(out_dir),
        "weights_file": weights_path.name,
        "hub_url": hub_url,
        "torch": torch.__version__,
    }


@app.function(
    gpu="RTX-3090",
    image=image,
    timeout=1200,
)
def train(
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    image_size: int = 128,
    max_train: int = 498,
    max_test: int = 500,
    dataset_id: str = DEFAULT_DATASET,
    seed: int = 0,
    push_to_hub: bool = False,
    hub_repo: str = "theoriclabs/not-hotdog-cnn",
) -> dict:
    """Train only. No Hugging Face token required."""
    return _train_impl(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        image_size=image_size,
        max_train=max_train,
        max_test=max_test,
        dataset_id=dataset_id,
        seed=seed,
        push_to_hub=push_to_hub,
        hub_repo=hub_repo,
    )


@app.function(
    gpu="RTX-3090",
    image=image,
    timeout=1200,
    secrets=[hf_secret],
)
def train_and_push(
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    image_size: int = 128,
    max_train: int = 498,
    max_test: int = 500,
    dataset_id: str = DEFAULT_DATASET,
    seed: int = 0,
    push_to_hub: bool = True,
    hub_repo: str = "theoriclabs/not-hotdog-cnn",
) -> dict:
    """Same train, then upload ``model.pt`` using the stored ``hf`` secret."""
    return _train_impl(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        image_size=image_size,
        max_train=max_train,
        max_test=max_test,
        dataset_id=dataset_id,
        seed=seed,
        push_to_hub=push_to_hub,
        hub_repo=hub_repo,
    )


if __name__ == "__main__":
    with app.run():
        print(train.remote(epochs=1, max_train=64, max_test=64))
