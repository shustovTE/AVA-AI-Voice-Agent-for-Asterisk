"""Public model publication must remain safe and readable across deployment UIDs."""

import errno
import hashlib
import io
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile

import pytest

from src.core import model_fetch


pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions")
PAYLOAD = b"public model fixture"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()
URL = "https://example.invalid/public-model.onnx"
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("umask", [0o000, 0o077])
def test_download_is_private_until_verified_then_readable_across_uids(tmp_path, umask):
    destination = tmp_path / "model.onnx"
    destination.write_bytes(b"previous model")

    class ObservedResponse(io.BytesIO):
        def read(self, size=-1):
            # While downloading, readers still see the complete previous model.
            assert destination.read_bytes() == b"previous model"
            [temporary] = tmp_path.glob(".model.*.part")
            assert stat.S_IMODE(temporary.stat().st_mode) == 0o600
            return super().read(size)

    previous_umask = os.umask(umask)
    try:
        model_fetch.download_file(
            str(destination), url=URL, sha256=DIGEST,
            opener=lambda *args, **kwargs: ObservedResponse(PAYLOAD),
        )
    finally:
        os.umask(previous_umask)

    assert destination.read_bytes() == PAYLOAD
    # Owner write, cross-UID read, no group/other write or executable bits.
    assert stat.S_IMODE(destination.stat().st_mode) == 0o644
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("failure", ["checksum", "interrupted"])
def test_failed_download_preserves_existing_bytes_and_permissions(tmp_path, failure):
    destination = tmp_path / "model.onnx"
    destination.write_bytes(b"previous model")
    destination.chmod(0o600)

    class BrokenResponse(io.BytesIO):
        def read(self, size=-1):
            if failure == "interrupted" and self.tell():
                raise OSError("connection lost")
            return super().read(size)

    with pytest.raises(RuntimeError, match="sha256|connection lost"):
        model_fetch.download_file(
            str(destination), url=URL, sha256=DIGEST,
            opener=lambda *args, **kwargs: BrokenResponse(b"incomplete model"),
        )

    assert destination.read_bytes() == b"previous model"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert list(tmp_path.iterdir()) == [destination]


def test_existing_custom_model_is_not_republished_or_made_public(tmp_path):
    destination = tmp_path / "private-model.onnx"
    destination.write_bytes(b"operator-provisioned model")
    destination.chmod(0o600)

    def unexpected_download(*args, **kwargs):
        pytest.fail("An existing custom model must not be downloaded again")

    assert model_fetch.ensure_file(
        str(destination), url=URL, sha256=DIGEST, version="fixture",
        opener=unexpected_download,
    ) == str(destination)
    assert destination.read_bytes() == b"operator-provisioned model"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600


@pytest.mark.parametrize("script_name", ["fetch_silero_vad.sh", "fetch_smart_turn.sh"])
@pytest.mark.parametrize("valid", [True, False], ids=["verified", "bad-checksum"])
def test_host_scripts_publish_only_verified_public_models(tmp_path, script_name, valid):
    if not all(shutil.which(tool) for tool in ("bash", "sha256sum", "mktemp")):
        pytest.skip("Host fetch scripts require bash, sha256sum, and mktemp")

    # Keep the real checksum verifier and publication code. Only replace the
    # pinned digest in a temporary script copy and supply offline curl bytes.
    script, count = re.subn(
        r'^SHA256="[0-9a-f]{64}"$', f'SHA256="{DIGEST}"',
        (ROOT / "scripts" / script_name).read_text(), flags=re.MULTILINE,
    )
    assert count == 1
    script_path = tmp_path / script_name
    script_path.write_text(script)
    payload = tmp_path / "payload"
    payload.write_bytes(PAYLOAD if valid else b"corrupt download")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    curl = fake_bin / "curl"
    curl.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\n'
        'while [[ $# -gt 0 ]]; do\n'
        '  if [[ "$1" == "-o" ]]; then\n'
        '    cp "$MODEL_TEST_PAYLOAD" "$2"\n'
        '    exit 0\n'
        '  fi\n'
        '  shift\n'
        'done\nexit 2\n'
    )
    curl.chmod(0o755)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    destination = model_dir / "model.onnx"
    destination.write_bytes(b"previous model")
    destination.chmod(0o600)

    result = subprocess.run(
        ["bash", "-c", 'umask 077; exec bash "$1" "$2"',
         "fetch-test", str(script_path), str(destination)],
        env={**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
             "MODEL_TEST_PAYLOAD": str(payload)},
        capture_output=True, text=True, timeout=10,
    )

    if valid:
        assert result.returncode == 0, result.stderr
        assert destination.read_bytes() == PAYLOAD
        assert stat.S_IMODE(destination.stat().st_mode) == 0o644
    else:
        assert result.returncode != 0
        assert "Checksum mismatch" in result.stderr
        assert destination.read_bytes() == b"previous model"
        assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert list(model_dir.iterdir()) == [destination]


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() != 0,
    reason="A different-UID subprocess requires root; other tests check mode bits",
)
def test_different_uid_can_read_public_model_but_cannot_write_it():
    # A traversable directory outside pytest's private temp tree models the
    # host bind-mount source. No Docker daemon or network is needed.
    with tempfile.TemporaryDirectory(prefix="public-model-", dir="/tmp") as directory:
        Path(directory).chmod(0o755)
        destination = Path(directory) / "model.onnx"
        model_fetch.download_file(
            str(destination), url=URL, sha256=DIGEST,
            opener=lambda *args, **kwargs: io.BytesIO(PAYLOAD),
        )
        probe = """
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
assert os.getuid() != path.stat().st_uid
if sys.argv[2] == "public":
    assert path.read_bytes() == b"public model fixture"
else:
    try:
        path.read_bytes()
    except PermissionError:
        pass
    else:
        raise AssertionError("0600 must deny the different UID")
try:
    with path.open("ab") as handle:
        handle.write(b"tampered")
except PermissionError:
    pass
else:
    raise AssertionError("A different UID must not be able to write")
"""
        for policy in ("public", "owner-only"):
            if policy == "owner-only":
                destination.chmod(0o600)
            try:
                result = subprocess.run(
                    [os.path.realpath(sys.executable), "-I", "-c", probe,
                     str(destination), policy],
                    user=65534, group=65534, extra_groups=(), cwd="/",
                    capture_output=True, text=True, timeout=10,
                )
            except PermissionError as exc:
                if exc.errno == errno.EPERM:
                    pytest.skip("This environment does not permit changing process UID/GID")
                raise
            assert result.returncode == 0, result.stderr
        assert destination.read_bytes() == PAYLOAD
