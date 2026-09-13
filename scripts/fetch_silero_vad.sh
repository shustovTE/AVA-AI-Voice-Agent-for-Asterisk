#!/usr/bin/env bash
# Fetch the pinned Silero VAD model for the AI Engine (vad.silero_enabled).
#
# The engine downloads the same file itself on first start when
# vad.silero_auto_download is true; use this script for hosts without outbound
# access from the container, or to pre-seed models/ before a restart.
#
# Usage: scripts/fetch_silero_vad.sh [destination]   (default: models/vad/silero_vad.onnx)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
DEST="${1:-${ROOT_DIR}/models/vad/silero_vad.onnx}"

# Keep these three lines in step with src/core/silero_vad.py.
VERSION="6.2.1"
URL="https://raw.githubusercontent.com/snakers4/silero-vad/v${VERSION}/src/silero_vad/data/silero_vad.onnx"
SHA256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"

if [[ -s "${DEST}" ]] && echo "${SHA256}  ${DEST}" | sha256sum -c --status 2>/dev/null; then
  echo "✅ Silero VAD v${VERSION} already present: ${DEST}"
  exit 0
fi

mkdir -p "$(dirname "${DEST}")"
TMP="$(mktemp "$(dirname "${DEST}")/.silero_vad.XXXXXX")"
trap 'rm -f "${TMP}"' EXIT

echo "⬇️  Downloading Silero VAD v${VERSION}"
curl -fsSL --retry 3 "${URL}" -o "${TMP}"
if ! echo "${SHA256}  ${TMP}" | sha256sum -c --status; then
  echo "❌ Checksum mismatch for ${URL}; the file was discarded" >&2
  exit 1
fi
chmod 0644 "${TMP}"
mv "${TMP}" "${DEST}"
trap - EXIT
echo "✅ Silero VAD v${VERSION} saved to ${DEST}"
