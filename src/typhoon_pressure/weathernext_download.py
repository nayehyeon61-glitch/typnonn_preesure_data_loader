from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen


@dataclass(frozen=True)
class URLCheckpointDownloader:
    """Download a user-supplied WeatherNext checkpoint URL once and cache it.

    The resolver intentionally does not hard-code a vendor URL. This adapter is
    useful for public HTTPS checkpoint URLs; authenticated/cloud-specific
    downloaders can implement the same ``download`` protocol separately.
    """

    url: str
    cache_dir: str | Path = "download/weathernext"
    filename: str | None = None

    def download(self, *, model_variant: str, release: str) -> Path:
        parsed = urlparse(self.url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("URLCheckpointDownloader supports only http/https URLs")

        source_name = Path(parsed.path).name
        filename = self.filename or source_name or "checkpoint.npz"
        destination_dir = Path(self.cache_dir).expanduser() / model_variant / release
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / filename
        if destination.is_file() and destination.stat().st_size > 0:
            return destination.resolve()

        with tempfile.NamedTemporaryFile(
            prefix=filename + ".",
            suffix=".part",
            dir=destination_dir,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            try:
                with urlopen(self.url) as response:
                    shutil.copyfileobj(response, temporary)
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise

        if temporary_path.stat().st_size == 0:
            temporary_path.unlink(missing_ok=True)
            raise IOError(f"Downloaded WeatherNext checkpoint is empty: {self.url}")
        temporary_path.replace(destination)
        return destination.resolve()
