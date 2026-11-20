#!/usr/bin/env bash
set -euo pipefail

image=ghcr.io/miaai-lab/qwen3.8-27b-dgx-spark-tensorfold@sha256:043d192b9e1725f55c5439fc9672055de891433e5b31448ed8963918e3c460e9
target=Vontra/Qwen3.8-27B-MLX-4bit
draft=z-lab/Qwen3.8-27B-DFlash2
target_revision=22d8d538154e0e5b6f8dcb1fcc74a60a0798104c
draft_revision=50307d4c4cde6860d4eee73e2547cd786fe8e8a4
cache="$HOME/.cache/spark-qwen38-27b-recipe"
hub="$cache/huggingface/hub"
name=spark-recipe-qwen38-27b-tensorfold

download_model() {
  local id="$1" revision="$2"
  local refs="$hub/models--${id//\//--}/refs"
  HF_HOME="$cache/huggingface" HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    hf download "$id" --cache-dir "$hub" --revision "$revision"
  # TensorFold loads the offline model and its automatic drafter by repository ID.
  mkdir -p "$refs"
  printf '%s' "$revision" > "$refs/main"
}

require_pin() {
  local id="$1" revision="$2"
  local model_cache="$hub/models--${id//\//--}"
  test "$(cat "$model_cache/refs/main")" = "$revision"
  test -f "$model_cache/snapshots/$revision/config.json"
}

case "${1:-}" in
  download)
    mkdir -p "$cache/runtime" "$hub"
    docker pull --platform linux/arm64 "$image"
    download_model "$target" "$target_revision"
    download_model "$draft" "$draft_revision"
    ;;
  start)
    require_pin "$target" "$target_revision"
    require_pin "$draft" "$draft_revision"
    docker run -d --name "$name" --network host --ipc host --gpus all \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      --mount "type=bind,source=$cache/runtime,target=/cache" \
      --mount "type=bind,source=$cache/huggingface,target=/root/.cache/huggingface,readonly" \
      --env HF_HUB_OFFLINE=1 --env HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
      --env TENSORFOLD_KV_DTYPE=fp8 --env TENSORFOLD_KV_POOL_GIB=32 \
      --env TENSORFOLD_MEMORY_RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-8}" \
      --env TENSORFOLD_MAX_IMAGES=50 --env TENSORFOLD_IMAGE_TOKENS=16384 \
      --env TENSORFOLD_VIDEO_TOKENS=16384 --env TENSORFOLD_NO_UPDATE_CHECK=1 \
      "$image" tensorfold serve "$target" --host 127.0.0.1 --port 8888 \
      --name Qwen3.8-27B --parallel 8 --context 262144 \
      --temperature 1.0 --top-p 0.95 --top-k 20 --prefill-fp8 --vision --thinking
    ;;
  stop)
    docker stop "$name"
    docker rm "$name"
    ;;
  *)
    echo "Usage: $0 {download|start|stop}" >&2
    exit 2
    ;;
esac
