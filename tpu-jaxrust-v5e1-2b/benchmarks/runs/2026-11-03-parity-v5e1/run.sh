set -x
cd ~/tpu-jaxrust-v5e1-2b
python3.13 tools/parity.py jax --out parity/jax.json > parity/jax.log 2>&1
python3.13 tools/parity.py rust --out parity/rust.json > parity/rust.log 2>&1
python3.13 tools/parity.py compare parity/jax.json parity/rust.json > parity/compare.txt 2>&1
echo ALLDONE
