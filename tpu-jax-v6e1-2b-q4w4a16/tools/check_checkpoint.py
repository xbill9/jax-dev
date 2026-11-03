"""Range-read every safetensors header of a Hub checkpoint and apply ports/gemma4/jax_e_loader.py's rules.

Stdlib only, no weights downloaded, no JAX needed:

    python3 tools/check_checkpoint.py <repo> <revision>
"""
import collections
import json
import struct
import sys
import urllib.error
import urllib.request

repo, rev = sys.argv[1], sys.argv[2]
base = f"https://huggingface.co/{repo}/resolve/{rev}/"
try:
    files = sorted(set(json.load(urllib.request.urlopen(base + "model.safetensors.index.json"))["weight_map"].values()))
except urllib.error.HTTPError as e:  # single-file checkpoint, e.g. google/gemma-4-E2B-it
    if e.code != 404:
        raise
    files = ["model.safetensors"]
hdr = {}
for f in files:
    r = urllib.request.Request(base + f, headers={"Range": "bytes=0-7"})
    n = struct.unpack("<Q", urllib.request.urlopen(r).read())[0]
    r = urllib.request.Request(base + f, headers={"Range": f"bytes=8-{7+n}"})
    h = json.loads(urllib.request.urlopen(r).read()); h.pop("__metadata__", None); hdr.update(h)
cfg = json.load(urllib.request.urlopen(base + "config.json"))
tc = cfg.get("text_config", cfg)
L, shared = tc["num_hidden_layers"], tc["num_hidden_layers"] - tc["num_kv_shared_layers"]
P = "model.language_model."
bad, packed_n, dense = [], 0, collections.Counter()
for k, v in hdr.items():
    if k.endswith(".weight_packed"):
        packed_n += 1
        s = hdr.get(k[:-7] + "_scale"); shp = hdr.get(k[:-7] + "_shape")
        if v["dtype"] != "I32": bad.append((k, "dtype", v["dtype"]))
        if s is None: bad.append((k, "no scale")); continue
        if len(v["shape"]) != 2 or len(s["shape"]) != 2: bad.append((k, "rank"))
        if s["shape"] != [v["shape"][0], v["shape"][1] // 4]: bad.append((k, "scale shape", s["shape"], v["shape"]))
        if s["dtype"] != "BF16": bad.append((k, "scale dtype", s["dtype"]))
for need in ("embed_tokens.weight", "norm.weight", "embed_tokens_per_layer.weight", "per_layer_projection_norm.weight"):
    if P + need not in hdr: bad.append(("missing", need))
for i in range(L):
    lp = f"{P}layers.{i}."
    projs = ["self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
             "per_layer_input_gate", "per_layer_projection"]
    if i < shared: projs += ["self_attn.k_proj", "self_attn.v_proj"]
    for p in projs:
        if lp + p + ".weight_packed" not in hdr:
            if lp + p + ".weight" in hdr: dense[p] += 1
            else: bad.append(("missing", lp + p))
pm = P + "per_layer_model_projection"
print(f"{repo}@{rev[:8]}: {len(hdr)} tensors, {packed_n} packed, layers {L}, kv-shared from {shared}")
print(" per_layer_model_projection:", "packed" if pm + ".weight_packed" in hdr else "dense" if pm + ".weight" in hdr else "MISSING")
print(" embed_tokens:", hdr[P + "embed_tokens.weight"]["dtype"], hdr[P + "embed_tokens.weight"]["shape"])
print(" dense fallbacks:", dict(dense) or "none")
print(" non-text tensors:", sum(1 for k in hdr if not k.startswith(P)))
print(" problems:", bad[:10] or "none", len(bad))
