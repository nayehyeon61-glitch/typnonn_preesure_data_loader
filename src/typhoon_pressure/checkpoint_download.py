"""Download official WeatherNext checkpoints from DeepMind's public bucket."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from urllib.parse import quote
from urllib.request import Request, urlopen

BUCKET = "dm_graphcast"
PREFIX = "weathernext2/params"
SUPPORTED_RELEASE = "v0.3.0"


@dataclass(frozen=True)
class OfficialCheckpoint:
    filename: str
    model_variant: str
    resolution_degrees: float

    @property
    def model_id(self) -> str:
        return self.filename.removesuffix(".npz")

    @property
    def object_path(self) -> str:
        return f"{PREFIX}/{self.filename}"

    @property
    def source_url(self) -> str:
        encoded = quote(self.object_path, safe="/")
        return f"https://storage.googleapis.com/{BUCKET}/{encoded}"


def _official_checkpoints() -> tuple[OfficialCheckpoint, ...]:
    checkpoints = [
        OfficialCheckpoint(
            f"WeatherNext2_<2025_model{member}.npz", "WeatherNext2", 0.25
        )
        for member in range(1, 5)
    ]
    for split in ("2025", "2024", "2023"):
        checkpoints.extend(
            OfficialCheckpoint(
                f"WeatherNextCyclones_<{split}_model{member}.npz",
                "WeatherNextCyclones",
                0.25,
            )
            for member in range(1, 5)
        )
    checkpoints.extend(
        [
            OfficialCheckpoint(
                "WeatherNextCyclones_Mini_<2024.npz",
                "WeatherNextCyclones_Mini",
                1.0,
            ),
            OfficialCheckpoint(
                "WeatherNextCyclones_Mini_<2023.npz",
                "WeatherNextCyclones_Mini",
                1.0,
            ),
        ]
    )
    return tuple(checkpoints)


OFFICIAL_CHECKPOINTS = _official_checkpoints()
_BY_FILENAME = {item.filename.lower(): item for item in OFFICIAL_CHECKPOINTS}
_ALIASES = {
    "weathernext2": "WeatherNext2_<2025_model1.npz",
    "weather-next2": "WeatherNext2_<2025_model1.npz",
    "cyclones": "WeatherNextCyclones_<2025_model1.npz",
    "cyclone": "WeatherNextCyclones_<2025_model1.npz",
    "mini": "WeatherNextCyclones_Mini_<2024.npz",
}


def resolve_official_checkpoint(name: str) -> OfficialCheckpoint:
    value = name.strip()
    filename = _ALIASES.get(value.lower(), value)
    if not filename.lower().endswith(".npz"):
        filename += ".npz"
    try:
        return _BY_FILENAME[filename.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown official WeatherNext checkpoint {name!r}; "
            "run with --list to see supported files"
        ) from exc


def _write_metadata(path: Path, spec: OfficialCheckpoint) -> Path:
    metadata_path = path.with_suffix(".metadata.json")
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".part")
    metadata = {
        "checkpoint_format": "weathernext.weathernext2.fgn.CheckPoint",
        "checkpoint_kind": "official_pretrained",
        "official_pretrained": True,
        "weathernext_release": SUPPORTED_RELEASE,
        "model_name": spec.model_id,
        "model_variant": spec.model_variant,
        "resolution_degrees": spec.resolution_degrees,
        "source_bucket": f"gs://{BUCKET}",
        "source_object": spec.object_path,
        "source_url": spec.source_url,
        "inference_ready": True,
    }
    temporary.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(metadata_path)
    return metadata_path


def download_official_checkpoint(
    name: str,
    output_dir: str | Path = "download",
    *,
    force: bool = False,
    open_url: Callable[..., BinaryIO] = urlopen,
) -> tuple[Path, Path]:
    """Atomically download one known checkpoint and write its provenance sidecar."""
    spec = resolve_official_checkpoint(name)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / spec.filename
    if destination.exists() and not force:
        if destination.stat().st_size == 0:
            raise ValueError(f"Existing checkpoint is empty: {destination}")
        return destination, _write_metadata(destination, spec)

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(spec.source_url, headers={"User-Agent": "typhoon-pressure-loader/0.4"})
    written = 0
    try:
        with open_url(request, timeout=60) as source, temporary.open("wb") as target:
            while chunk := source.read(8 * 1024 * 1024):
                target.write(chunk)
                written += len(chunk)
            expected = source.headers.get("Content-Length")
        if expected is not None and written != int(expected):
            raise OSError(
                f"Incomplete checkpoint download: expected {expected} bytes, got {written}"
            )
        if written == 0:
            raise OSError("Downloaded checkpoint is empty")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    metadata_path = _write_metadata(destination, spec)
    return destination, metadata_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download an official WeatherNext checkpoint to ./download"
    )
    parser.add_argument("--model", help="Official filename or alias: weathernext2, cyclones, mini")
    parser.add_argument("--output-dir", default="download")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true", dest="list_models")
    args = parser.parse_args(argv)

    if args.list_models:
        for spec in OFFICIAL_CHECKPOINTS:
            print(f"{spec.filename}\t{spec.model_variant}\t{spec.resolution_degrees}deg")
        return 0
    if not args.model:
        parser.error("--model is required unless --list is used")
    checkpoint_path, metadata_path = download_official_checkpoint(
        args.model, args.output_dir, force=args.force
    )
    print(f"checkpoint={checkpoint_path}")
    print(f"metadata={metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
