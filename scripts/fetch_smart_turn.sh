#!/usr/bin/env bash
# Fetch the pinned Smart Turn v3 model for the AI Engine (vad.smart_turn_enabled).
#
# The engine downloads the same file itself on first start when
# vad.smart_turn_auto_download is true; use this script for hosts without
# outbound access from the container, or to pre-seed models/ before a restart.
#
# Usage: scripts/fetch_smart_turn.sh [destination]   (default: models/turn/smart-turn-v3.2-cpu.onnx)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
DEST="${1:-${ROOT_DIR}/models/turn/smart-turn-v3.2-cpu.onnx}"

# Keep these three lines in step with src/core/smart_turn.py.
VERSION="3.2"
URL="https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx"
SHA256="2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f"

if [[ -s "${DEST}" ]] && echo "${SHA256}  ${DEST}" | sha256sum -c --status 2>/dev/null; then
  echo "✅ Smart Turn v${VERSION} already present: ${DEST}"
  exit 0
fi

mkdir -p "$(dirname "${DEST}")"
TMP="$(mktemp "$(dirname "${DEST}")/.smart_turn.XXXXXX")"
trap 'rm -f "${TMP}"' EXIT

echo "⬇️  Downloading Smart Turn v${VERSION} (about 8 MB)"
curl -fsSL --retry 3 "${URL}" -o "${TMP}"
if ! echo "${SHA256}  ${TMP}" | sha256sum -c --status; then
  echo "❌ Checksum mismatch for ${URL}; the file was discarded" >&2
  exit 1
fi
chmod 0644 "${TMP}"
mv "${TMP}" "${DEST}"
trap - EXIT
echo "✅ Smart Turn v${VERSION} saved to ${DEST}"
