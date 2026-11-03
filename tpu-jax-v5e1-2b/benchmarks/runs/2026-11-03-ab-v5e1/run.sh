set -x
cd ~/tpu-jax-v5e1-2b
python3.13 tools/ab_bench.py --model google/gemma-4-E2B-it --quant-mode fp16 --out ab/bf16.json > ab/bf16.log 2>&1
python3.13 tools/ab_bench.py --model xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text --quant-mode w4a16 --out ab/q4w4a16.json > ab/q4w4a16.log 2>&1
python3.13 tools/ab_bench.py --model google/gemma-4-E2B-it-qat-w4a16-ct --quant-mode w4a16 --out ab/w4a16.json > ab/w4a16.log 2>&1
echo ALLDONE
