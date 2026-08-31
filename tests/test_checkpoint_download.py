import io
import json

from typhoon_pressure.checkpoint_download import (
    download_official_checkpoint,
    resolve_official_checkpoint,
)


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_alias_resolves_to_official_checkpoint():
    spec = resolve_official_checkpoint("mini")
    assert spec.filename == "WeatherNextCyclones_Mini_<2024.npz"
    assert spec.model_variant == "WeatherNextCyclones_Mini"
    assert "%3C2024" in spec.source_url


def test_download_writes_checkpoint_and_inference_metadata(tmp_path):
    payload = b"official-checkpoint"
    requests = []

    def open_url(request, timeout):
        requests.append((request.full_url, timeout))
        return FakeResponse(payload)

    checkpoint, metadata = download_official_checkpoint(
        "mini", tmp_path, open_url=open_url
    )
    assert checkpoint.read_bytes() == payload
    details = json.loads(metadata.read_text(encoding="utf-8"))
    assert details["checkpoint_kind"] == "official_pretrained"
    assert details["model_variant"] == "WeatherNextCyclones_Mini"
    assert requests[0][1] == 60
