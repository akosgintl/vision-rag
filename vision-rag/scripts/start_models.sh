#!/usr/bin/env bash
# ============================================================
# Start all three vLLM model servers on a single GPU
# Run this script before starting the FastAPI proxy
# ============================================================

set -euo pipefail

echo "═══════════════════════════════════════════════════════"
echo "  Multi-Model vLLM Launcher"
echo "  GPU: Single NVIDIA A100 80GB"
echo "═══════════════════════════════════════════════════════"

# ── Configuration ────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0
MODEL_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

# ── Model 1: ColPali (Retrieval) ─────────────────────────
echo ""
echo "[1/3] Starting ColPali (TomoroAI/tomoro-ai-colqwen3-embed-8b-awq) on port 8001..."
vllm serve TomoroAI/tomoro-ai-colqwen3-embed-8b-awq \
    --port 8001 \
    --gpu-memory-utilization 0.10 \
    --max-model-len 32768 \
    --dtype auto \
    --task embed \
    --trust-remote-code \
    --disable-log-requests &
PID_COLPALI=$!
echo "    PID: $PID_COLPALI"

# ── Model 2: Qwen3-VL (Visual Extraction) ───────────────
echo ""
echo "[2/3] Starting Qwen3-VL-2B-Instruct on port 8002..."
vllm serve Qwen/Qwen3-VL-2B-Instruct \
    --port 8002 \
    --gpu-memory-utilization 0.08 \
    --max-model-len 4096 \
    --dtype float16 \
    --limit-mm-per-prompt image=5 \
    --trust-remote-code \
    --disable-log-requests &
PID_QWEN3VL=$!
echo "    PID: $PID_QWEN3VL"

# ── Model 3: Qwen2.5-7B (Text Generation) ───────────────
echo ""
echo "[3/3] Starting Qwen2.5-7B-Instruct-AWQ on port 8003..."
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
    --port 8003 \
    --gpu-memory-utilization 0.18 \
    --max-model-len 8192 \
    --quantization awq \
    --trust-remote-code \
    --enable-chunked-prefill \
    --disable-log-requests &
PID_QWEN25=$!
echo "    PID: $PID_QWEN25"

# ── Wait for health ─────────────────────────────────────
echo ""
echo "Waiting for all models to be ready..."

wait_for_health() {
    local port=$1
    local name=$2
    local max_retries=120
    local i=0
    while [ $i -lt $max_retries ]; do
        if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "  ✓ $name (port $port) is healthy"
            return 0
        fi
        sleep 2
        i=$((i + 1))
    done
    echo "  ✗ $name (port $port) failed to start after ${max_retries}s"
    return 1
}

wait_for_health 8001 "ColPali"
wait_for_health 8002 "Qwen3-VL"
wait_for_health 8003 "Qwen2.5-7B"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  All models are ready!"
echo "  ColPali:     http://localhost:8001"
echo "  Qwen3-VL:    http://localhost:8002"
echo "  Qwen2.5-7B:  http://localhost:8003"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "  PIDs: $PID_COLPALI $PID_QWEN3VL $PID_QWEN25"
echo "  To stop: kill $PID_COLPALI $PID_QWEN3VL $PID_QWEN25"
echo ""

# ── Trap cleanup ─────────────────────────────────────────
cleanup() {
    echo "Shutting down models..."
    kill $PID_COLPALI $PID_QWEN3VL $PID_QWEN25 2>/dev/null
    wait
    echo "All models stopped."
}
trap cleanup SIGINT SIGTERM

# Keep script running
wait
