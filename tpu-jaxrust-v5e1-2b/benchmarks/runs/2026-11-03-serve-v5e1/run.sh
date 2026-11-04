# One v5e-1, three engines, one client (tools/serve_bench.py), one engine on the chip at a time.
set -x
export PJRT_PLUGIN=/usr/local/lib/python3.13/dist-packages/libtpu/libtpu.so
B=~/tpu-jaxrust-v5e1-2b/tools/serve_bench.py
O=~/serve; mkdir -p $O
REV=4adf15748de9e1a0097b6274cee9c15836c41c37
stop_server() {  # SIGTERM, then SIGKILL; return once nothing answers on :8000
  kill $1; for i in $(seq 1 30); do kill -0 $1 2>/dev/null || break; sleep 1; done
  kill -9 $1 2>/dev/null; while curl -sf localhost:8000/v1/models >/dev/null; do sleep 1; done; sleep 5
}
wait_up() { for i in $(seq 1 240); do curl -sf localhost:8000/v1/models >/dev/null && return 0; sleep 10; done; return 1; }

# 1. Rust engine
cd ~/tpu-jaxrust-v5e1-2b; python3.13 rust_openai_server.py --context 2048 > $O/rust.server.log 2>&1 &
RS=$!; wait_up && python3.13 $B run --label rust --out $O/rust.json --note "rust_openai_server.py --context 2048" > $O/rust.log 2>&1
stop_server $RS

# 2. Python JAX engine
cd ~/tpu-jax-v5e1-2b; python3.13 jax_openai_server.py --model google/gemma-4-E2B-it --quant-mode fp16 --max-model-len 2048 > $O/jax.server.log 2>&1 &
JS=$!; wait_up && python3.13 $B run --label jax --out $O/jax.json --note "jax_openai_server.py --quant-mode fp16 --max-model-len 2048" > $O/jax.log 2>&1
stop_server $JS

# 3. vLLM, the flags of ~/gemma4-dev/jev-tpu-v5e1/tpu/serve.sh without prefix caching
IMG=$(sudo docker inspect --format '{{index .RepoDigests 0}}' vllm/vllm-tpu:nightly); echo "image $IMG"
sudo docker run -d --name vllm --privileged --net=host --shm-size 10gb -v /dev/shm:/dev/shm \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 "$IMG" \
  vllm serve google/gemma-4-E2B-it --revision $REV --served-model-name google/gemma-4-E2B-it --host 127.0.0.1 --port 8000 \
  --max-model-len 2048 --max-num-seqs 16 --generation-config vllm \
  --limit-mm-per-prompt '{"image":0,"audio":0,"video":0}' --no-enable-prefix-caching
wait_up && python3.13 $B run --label vllm --out $O/vllm.json --note "vllm $IMG, --max-model-len 2048 --max-num-seqs 16 --no-enable-prefix-caching" > $O/vllm.log 2>&1
sudo docker logs vllm > $O/vllm.server.log 2>&1; sudo docker rm -f vllm
python3.13 $B compare $O/vllm.json $O/jax.json $O/rust.json > $O/compare.txt 2>&1
echo ALLDONE
