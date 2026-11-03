//! Gemma 4 E2B in Rust: the config, the weights on the device, the forward pass emitted as
//! StableHLO, and the prefill/decode loop. No Python here.
//!
//! The graph follows `ports/gemma4/jax_e_model.py` of the jax-dev engine operation for
//! operation, including its element types: JAX promotes bf16 to f32 wherever the two meet,
//! so after the first RoPE the activations are f32 while every weight stays bf16 on the chip
//! and is converted inside the matmul that reads it. `hlo.rs` applies the same promotion.
//!
//! One sequence at a time. The KV cache holds `context` positions per KV-owning layer and is
//! donated to each program, which writes into it in place. Position p lives at slot p: the
//! prompt is right-padded to a 128 bucket for the prefill, and decoding starts at the prompt's
//! real length, overwriting the pad slots, so slot and position never diverge.
use crate::hlo::{Builder, Cmp, DType, V};
use crate::pjrt::{Buffer, Exe, Result};
use crate::weights::Checkpoint;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

pub const BUCKET: i64 = 128;
const MASK_MIN: f64 = -1e9;
const PREFIX: &str = "model.language_model.";

#[derive(Clone, Debug)]
pub struct Config {
    pub vocab: i64,
    pub hidden: i64,
    pub layers: usize,
    pub heads: i64,
    pub kv_heads: i64,
    pub head_dim: i64,
    pub global_head_dim: i64,
    pub eps: f64,
    pub rope_local: f64,
    pub rope_global: f64,
    pub partial_rotary: f64,
    pub final_softcap: f64,
    pub kv_shared: usize,
    pub window: i64,
    pub sliding: Vec<bool>,
    pub ple_dim: i64,
}

impl Config {
    pub fn from_json(v: &Value) -> Result<Config> {
        let t = v.get("text_config").unwrap_or(v);
        let i = |k: &str| -> Result<i64> {
            t.get(k)
                .and_then(Value::as_i64)
                .ok_or_else(|| format!("config.json has no integer {k}"))
        };
        let f = |k: &str, d: f64| t.get(k).and_then(Value::as_f64).unwrap_or(d);
        let layers = i("num_hidden_layers")? as usize;
        let sliding: Vec<bool> = match t.get("layer_types").and_then(Value::as_array) {
            Some(a) => a
                .iter()
                .map(|x| x.as_str() == Some("sliding_attention"))
                .collect(),
            None => (0..layers).map(|i| i % 5 != 4).collect(),
        };
        let rope = t.get("rope_parameters");
        let theta = |kind: &str, d: f64| {
            rope.and_then(|r| r.get(kind))
                .and_then(|r| r.get("rope_theta"))
                .and_then(Value::as_f64)
                .unwrap_or(d)
        };
        let partial = rope
            .and_then(|r| r.get("full_attention"))
            .and_then(|r| r.get("partial_rotary_factor"))
            .and_then(Value::as_f64)
            .unwrap_or(0.25);
        let kv_heads = i("num_key_value_heads")?;
        let cfg = Config {
            vocab: i("vocab_size")?,
            hidden: i("hidden_size")?,
            layers,
            heads: i("num_attention_heads")?,
            kv_heads,
            head_dim: i("head_dim")?,
            global_head_dim: t
                .get("global_head_dim")
                .and_then(Value::as_i64)
                .unwrap_or(i("head_dim")?),
            eps: f("rms_norm_eps", 1e-6),
            rope_local: theta("sliding_attention", 10000.0),
            rope_global: theta("full_attention", 1000000.0),
            partial_rotary: partial,
            final_softcap: f("final_logit_softcapping", 0.0),
            kv_shared: i("num_kv_shared_layers")? as usize,
            window: i("sliding_window")?,
            sliding,
            ple_dim: i("hidden_size_per_layer_input")?,
        };
        if t.get("attention_k_eq_v").and_then(Value::as_bool) == Some(true) {
            return Err("attention_k_eq_v checkpoints are not supported".into());
        }
        if cfg.sliding.len() != layers {
            return Err("layer_types length differs from num_hidden_layers".into());
        }
        Ok(cfg)
    }

    pub fn first_shared(&self) -> usize {
        self.layers - self.kv_shared
    }

    /// The KV-owning layer each layer reads: itself, or for a shared layer the last layer of
    /// the same attention type before the shared block.
    pub fn share_map(&self) -> Vec<usize> {
        let first = self.first_shared();
        let mut last = HashMap::new();
        for i in 0..first {
            last.insert(self.sliding[i], i);
        }
        (0..self.layers)
            .map(|i| if i < first { i } else { last[&self.sliding[i]] })
            .collect()
    }

    fn dh(&self, layer: usize) -> i64 {
        if self.sliding[layer] {
            self.head_dim
        } else {
            self.global_head_dim
        }
    }
}

/// Names of one layer's tensors, in the order they are program parameters.
fn layer_names(cfg: &Config, i: usize) -> Vec<String> {
    let p = format!("{PREFIX}layers.{i}.");
    let mut n = vec![
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "per_layer_input_gate.weight",
        "per_layer_projection.weight",
        "post_per_layer_input_norm.weight",
        "layer_scalar",
    ];
    if i < cfg.first_shared() {
        n.extend([
            "self_attn.k_proj.weight",
            "self_attn.k_norm.weight",
            "self_attn.v_proj.weight",
        ]);
    }
    n.into_iter().map(|s| format!("{p}{s}")).collect()
}

const GLOBAL_NAMES: [&str; 5] = [
    "embed_tokens.weight",
    "embed_tokens_per_layer.weight",
    "per_layer_model_projection.weight",
    "per_layer_projection_norm.weight",
    "norm.weight",
];

/// Every weight, in parameter order, with its shape.
fn weight_list(cfg: &Config) -> Vec<String> {
    let mut v: Vec<String> = GLOBAL_NAMES
        .iter()
        .map(|s| format!("{PREFIX}{s}"))
        .collect();
    for i in 0..cfg.layers {
        v.extend(layer_names(cfg, i));
    }
    v
}

/// The shape of every weight, in parameter order, read from the checkpoint's headers.
pub fn weight_shapes(dir: &Path, cfg: &Config) -> Result<Vec<Vec<i64>>> {
    let ck = Checkpoint::open(dir)?;
    weight_list(cfg).iter().map(|n| ck.shape(n)).collect()
}

/// The weights as graph values, looked up by short name.
struct W {
    by_name: HashMap<String, V>,
}

impl W {
    fn g(&self, name: &str) -> &V {
        &self.by_name[&format!("{PREFIX}{name}")]
    }
    fn l(&self, i: usize, name: &str) -> &V {
        &self.by_name[&format!("{PREFIX}layers.{i}.{name}")]
    }
}

fn declare_weights(b: &mut Builder, names: &[String], shapes: &[Vec<i64>]) -> W {
    let mut by_name = HashMap::new();
    for (n, s) in names.iter().zip(shapes) {
        by_name.insert(n.clone(), b.param(s, DType::BF16, None));
    }
    W { by_name }
}

// --- graph pieces, each the JAX function of the same name -------------------------------------

fn rms_norm(b: &mut Builder, x: &V, w: Option<&V>, eps: f64) -> V {
    let dt = x.dt;
    let last = x.rank() as i64 - 1;
    let d = x.dim(-1) as f64;
    let xf = b.convert(x, DType::F32);
    let sq = b.mul(&xf, &xf);
    let sum = b.reduce_sum(&sq, &[last]);
    let var = b.scale(&sum, 1.0 / d);
    let var = b.add_scalar(&var, eps);
    let r = b.rsqrt(&var);
    let mut keep: Vec<i64> = x.shape.clone();
    *keep.last_mut().unwrap() = 1;
    let r = b.reshape(&r, &keep);
    let n = b.mul(&xf, &r);
    let n = b.convert(&n, dt);
    match w {
        Some(w) => b.mul(&n, w),
        None => n,
    }
}

/// jax.nn.gelu(x, approximate=True).
fn gelu_tanh(b: &mut Builder, x: &V) -> V {
    let x2 = b.mul(x, x);
    let x3 = b.mul(&x2, x);
    let t = b.scale(&x3, 0.044715);
    let inner = b.add(x, &t);
    let inner = b.scale(&inner, (2.0 / std::f64::consts::PI).sqrt());
    let th = b.tanh(&inner);
    let one = b.add_scalar(&th, 1.0);
    let cdf = b.scale(&one, 0.5);
    b.mul(x, &cdf)
}

fn rotate_half(b: &mut Builder, x: &V) -> V {
    let r = x.rank();
    let d = x.dim(-1);
    let mut s1 = vec![0i64; r];
    let mut l1 = x.shape.clone();
    l1[r - 1] = d / 2;
    let x1 = b.slice(x, &s1, &l1);
    s1[r - 1] = d / 2;
    let x2 = b.slice(x, &s1, &x.shape.clone());
    let nx2 = b.neg(&x2);
    b.concat(&[&nx2, &x1], r - 1)
}

/// cos/sin tables [S, d] for positions `pos` [S] (s32). Frequencies past `partial` of the
/// pairs are zero, so those channels pass through as cos 1, sin 0: the JAX engine's masked
/// table, computed exactly rather than masked.
fn rope_tables(b: &mut Builder, pos: &V, d: i64, theta: f64, partial: f64) -> (V, V) {
    let half = (d / 2) as usize;
    let n_rot = if partial < 1.0 {
        (partial * d as f64) as usize / 2
    } else {
        half
    };
    let inv: Vec<f32> = (0..half)
        .map(|k| {
            if k < n_rot {
                1.0f32 / (theta as f32).powf((2 * k) as f32 / d as f32)
            } else {
                0.0
            }
        })
        .collect();
    let s = pos.shape[0];
    let inv = b.f32_vec(&inv);
    let p = b.convert(pos, DType::F32);
    let p = b.broadcast(&p, &[s, half as i64], &[0]);
    let invb = b.broadcast(&inv, &[s, half as i64], &[1]);
    let freqs = b.mul(&p, &invb);
    let c = b.cos(&freqs);
    let sn = b.sin(&freqs);
    (b.concat(&[&c, &c], 1), b.concat(&[&sn, &sn], 1))
}

/// x [H, S, d] with tables [S, d].
fn apply_rope(b: &mut Builder, x: &V, cos: &V, sin: &V) -> V {
    let shape = x.shape.clone();
    let c = b.broadcast(cos, &shape, &[1, 2]);
    let s = b.broadcast(sin, &shape, &[1, 2]);
    let xc = b.mul(x, &c);
    let rh = rotate_half(b, x);
    let rs = b.mul(&rh, &s);
    b.add(&xc, &rs)
}

/// Heads-first: [S, n*d] -> [n, S, d].
fn heads(b: &mut Builder, x: &V, n: i64, d: i64) -> V {
    let s = x.dim(0);
    let r = b.reshape(x, &[s, n, d]);
    b.transpose(&r, &[1, 0, 2])
}

/// Grouped-query attention of q [H, S, d] over k, v [Hkv, T, d] with an additive mask [S, T].
fn attention(b: &mut Builder, cfg: &Config, q: &V, k: &V, v: &V, mask: &V) -> V {
    let (h, s, d) = (q.dim(0), q.dim(1), q.dim(2));
    let hkv = k.dim(0);
    let t = k.dim(1);
    let rep = h / hkv;
    // The cast eager_attention_jax makes when K/V are not the query's type.
    let k = b.convert(k, q.dt);
    let v = b.convert(v, q.dt);
    let qg = b.reshape(q, &[hkv, rep, s, d]);
    let scores = b.dot(&qg, &k, &[0], &[0], &[3], &[2]); // [Hkv, rep, S, T]
    let _ = cfg;
    let m = b.broadcast(mask, &[hkv, rep, s, t], &[2, 3]);
    let scores = b.add(&scores, &m);
    let sf = b.convert(&scores, DType::F32);
    let mx = b.reduce_max(&sf, &[3]);
    let mx = b.reshape(&mx, &[hkv, rep, s, 1]);
    let z = b.sub(&sf, &mx);
    let e = b.exp(&z);
    let sum = b.reduce_sum(&e, &[3]);
    let sum = b.reshape(&sum, &[hkv, rep, s, 1]);
    let p = b.div(&e, &sum);
    let p = b.convert(&p, q.dt);
    let out = b.dot(&p, &v, &[0], &[0], &[3], &[1]); // [Hkv, rep, S, d]
    let out = b.reshape(&out, &[h, s, d]);
    let out = b.transpose(&out, &[1, 0, 2]);
    b.reshape(&out, &[s, h * d])
}

/// Additive mask [S, T]: key slot j is visible to query position i when j <= i, j < `limit`
/// (the prompt's real length during prefill; unlimited when None) and, for a sliding layer,
/// j > i - window.
fn make_mask(b: &mut Builder, qpos: &V, t: i64, limit: Option<&V>, window: Option<i64>) -> V {
    let s = qpos.shape[0];
    let i = b.broadcast(qpos, &[s, t], &[0]);
    let j = b.iota(&[s, t], 1, DType::S32);
    let mut ok = b.compare(&j, &i, Cmp::Le);
    if let Some(l) = limit {
        let lb = b.broadcast(l, &[s, t], &[]);
        let c = b.compare(&j, &lb, Cmp::Lt);
        ok = b.and(&ok, &c);
    }
    if let Some(w) = window {
        let wv = b.splat(w as f64, &[s, t], DType::S32);
        let lo = b.sub(&i, &wv);
        let c = b.compare(&j, &lo, Cmp::Gt);
        ok = b.and(&ok, &c);
    }
    let zero = b.splat(0.0, &[s, t], DType::F32);
    let neg = b.splat(MASK_MIN, &[s, t], DType::F32);
    b.select(&ok, &zero, &neg)
}

/// What the forward pass needs from the program it is part of.
struct Step<'a> {
    ids: &'a V,           // [S] s32
    pos: &'a V,           // [S] s32, positions == cache slots
    start: &'a V,         // scalar s32: slot of ids[0]
    limit: Option<&'a V>, // scalar s32: keys at or past it are pad (prefill only)
    prefill: bool,
}

/// Runs the decoder over `step.ids`, writing K/V into `caches` (one (k, v) per KV-owning
/// layer, each [Hkv, T, d] bf16). Returns the final hidden states [S, D] and the new caches.
fn forward(
    b: &mut Builder,
    cfg: &Config,
    w: &W,
    step: &Step,
    caches: &[(V, V)],
) -> (V, Vec<(V, V)>) {
    let s = step.ids.shape[0];
    let dmodel = cfg.hidden;
    let l = cfg.layers as i64;
    let dp = cfg.ple_dim;

    let emb = b.gather_rows(w.g("embed_tokens.weight"), step.ids);
    let mut h = b.scale(&emb, (dmodel as f64).sqrt());

    // Per-layer embeddings: one [S, L, Dp] context, sliced per layer.
    let pe = b.gather_rows(w.g("embed_tokens_per_layer.weight"), step.ids);
    let pe = b.scale(&pe, (dp as f64).sqrt());
    let pe = b.reshape(&pe, &[s, l, dp]);
    let hs = b.scale(&h, 1.0 / (dmodel as f64).sqrt());
    let pp = b.linear(&hs, w.g("per_layer_model_projection.weight"));
    let pp = b.reshape(&pp, &[s, l, dp]);
    let pp = rms_norm(
        b,
        &pp,
        Some(w.g("per_layer_projection_norm.weight")),
        cfg.eps,
    );
    let ctx = b.add(&pp, &pe);
    let sq2 = b.splat(2f64.sqrt(), &ctx.shape.clone(), ctx.dt);
    let ple = b.div(&ctx, &sq2);

    let (cos_l, sin_l) = rope_tables(b, step.pos, cfg.head_dim, cfg.rope_local, 1.0);
    let (cos_g, sin_g) = rope_tables(
        b,
        step.pos,
        cfg.global_head_dim,
        cfg.rope_global,
        cfg.partial_rotary,
    );

    let t = caches[0].0.dim(1);
    let attend_len = if step.prefill { s } else { t };
    let mask_full = make_mask(b, step.pos, attend_len, step.limit, None);
    let mask_slide = make_mask(b, step.pos, attend_len, step.limit, Some(cfg.window));

    let share = cfg.share_map();
    let first = cfg.first_shared();
    let mut out_caches: Vec<(V, V)> = Vec::new();
    // What each KV-owning layer hands to the layers that share it: the fresh K/V during
    // prefill, the updated cache during decode — as in the JAX engine.
    let mut shared: HashMap<usize, (V, V)> = HashMap::new();
    let zero = b.scalar(0.0, DType::S32);

    for i in 0..cfg.layers {
        let sliding = cfg.sliding[i];
        let d = cfg.dh(i);
        let (cos, sin) = if sliding {
            (&cos_l, &sin_l)
        } else {
            (&cos_g, &sin_g)
        };
        let mask = if sliding { &mask_slide } else { &mask_full };

        let x = rms_norm(b, &h, Some(w.l(i, "input_layernorm.weight")), cfg.eps);
        let q = b.linear(&x, w.l(i, "self_attn.q_proj.weight"));
        let q = heads(b, &q, cfg.heads, d);
        let q = rms_norm(b, &q, Some(w.l(i, "self_attn.q_norm.weight")), cfg.eps);
        let q = apply_rope(b, &q, cos, sin);

        let (k, v) = if i < first {
            let k = b.linear(&x, w.l(i, "self_attn.k_proj.weight"));
            let v = b.linear(&x, w.l(i, "self_attn.v_proj.weight"));
            let k = heads(b, &k, cfg.kv_heads, d);
            let v = heads(b, &v, cfg.kv_heads, d);
            let k = rms_norm(b, &k, Some(w.l(i, "self_attn.k_norm.weight")), cfg.eps);
            let v = rms_norm(b, &v, None, cfg.eps);
            let k = apply_rope(b, &k, cos, sin);
            let (kc, vc) = &caches[i];
            let nk = b.dynamic_update_slice(kc, &k, &[&zero, step.start, &zero]);
            let nv = b.dynamic_update_slice(vc, &v, &[&zero, step.start, &zero]);
            out_caches.push((nk.clone(), nv.clone()));
            let kv = if step.prefill { (k, v) } else { (nk, nv) };
            shared.insert(i, kv.clone());
            kv
        } else {
            shared[&share[i]].clone()
        };

        let a = attention(b, cfg, &q, &k, &v, mask);
        let a = b.linear(&a, w.l(i, "self_attn.o_proj.weight"));
        let a = rms_norm(
            b,
            &a,
            Some(w.l(i, "post_attention_layernorm.weight")),
            cfg.eps,
        );
        h = b.add(&h, &a);

        let f = rms_norm(
            b,
            &h,
            Some(w.l(i, "pre_feedforward_layernorm.weight")),
            cfg.eps,
        );
        let g = b.linear(&f, w.l(i, "mlp.gate_proj.weight"));
        let u = b.linear(&f, w.l(i, "mlp.up_proj.weight"));
        let g = gelu_tanh(b, &g);
        let act = b.mul(&g, &u);
        let m = b.linear(&act, w.l(i, "mlp.down_proj.weight"));
        let m = rms_norm(
            b,
            &m,
            Some(w.l(i, "post_feedforward_layernorm.weight")),
            cfg.eps,
        );
        h = b.add(&h, &m);

        let gate = b.linear(&h, w.l(i, "per_layer_input_gate.weight"));
        let gate = gelu_tanh(b, &gate);
        let sl = b.slice(&ple, &[0, i as i64, 0], &[s, i as i64 + 1, dp]);
        let sl = b.reshape(&sl, &[s, dp]);
        let fused = b.mul(&gate, &sl);
        let back = b.linear(&fused, w.l(i, "per_layer_projection.weight"));
        let back = rms_norm(
            b,
            &back,
            Some(w.l(i, "post_per_layer_input_norm.weight")),
            cfg.eps,
        );
        h = b.add(&h, &back);

        h = b.mul(&h, w.l(i, "layer_scalar"));
    }
    let h = rms_norm(b, &h, Some(w.g("norm.weight")), cfg.eps);
    (h, out_caches)
}

/// Logits [V] f32 of one hidden row [1, D], softcapped.
fn logits(b: &mut Builder, cfg: &Config, w: &W, row: &V) -> V {
    let lg = b.linear(row, w.g("embed_tokens.weight"));
    let lg = b.reshape(&lg, &[cfg.vocab]);
    if cfg.final_softcap > 0.0 {
        let x = b.scale(&lg, 1.0 / cfg.final_softcap);
        let x = b.tanh(&x);
        b.scale(&x, cfg.final_softcap)
    } else {
        lg
    }
}

fn declare_caches(b: &mut Builder, cfg: &Config, t: i64, first_out: usize) -> Vec<(V, V)> {
    (0..cfg.first_shared())
        .map(|i| {
            let shape = [cfg.kv_heads, t, cfg.dh(i)];
            let k = b.param(&shape, DType::BF16, Some(first_out + 2 * i));
            let v = b.param(&shape, DType::BF16, Some(first_out + 2 * i + 1));
            (k, v)
        })
        .collect()
}

/// prefill(weights.., ids [S], len, caches..) -> (token, logits [V], caches..)
pub fn prefill_module(cfg: &Config, shapes: &[Vec<i64>], s: i64, t: i64) -> String {
    let mut b = Builder::new();
    let w = declare_weights(&mut b, &weight_list(cfg), shapes);
    let ids = b.param(&[s], DType::S32, None);
    let len = b.param(&[], DType::S32, None);
    let caches = declare_caches(&mut b, cfg, t, 2);
    let pos = b.iota(&[s], 0, DType::S32);
    let zero = b.scalar(0.0, DType::S32);
    let step = Step {
        ids: &ids,
        pos: &pos,
        start: &zero,
        limit: Some(&len),
        prefill: true,
    };
    let (h, new) = forward(&mut b, cfg, &w, &step, &caches);
    let one = b.scalar(1.0, DType::S32);
    let last = b.sub(&len, &one);
    let row = b.dynamic_slice(&h, &[&last, &zero], &[1, cfg.hidden]);
    let lg = logits(&mut b, cfg, &w, &row);
    let tok = b.argmax(&lg);
    let mut outs: Vec<&V> = vec![&tok, &lg];
    for (k, v) in &new {
        outs.push(k);
        outs.push(v);
    }
    b.finish(&outs)
}

/// decode(weights.., tok, pos, caches..) -> (token, caches..)
pub fn decode_module(cfg: &Config, shapes: &[Vec<i64>], t: i64) -> String {
    let mut b = Builder::new();
    let w = declare_weights(&mut b, &weight_list(cfg), shapes);
    let tok = b.param(&[], DType::S32, None);
    let pos = b.param(&[], DType::S32, None);
    let caches = declare_caches(&mut b, cfg, t, 1);
    let ids = b.reshape(&tok, &[1]);
    let posv = b.reshape(&pos, &[1]);
    let step = Step {
        ids: &ids,
        pos: &posv,
        start: &pos,
        limit: None,
        prefill: false,
    };
    let (h, new) = forward(&mut b, cfg, &w, &step, &caches);
    let lg = logits(&mut b, cfg, &w, &h);
    let next = b.argmax(&lg);
    let mut outs: Vec<&V> = vec![&next];
    for (k, v) in &new {
        outs.push(k);
        outs.push(v);
    }
    b.finish(&outs)
}

// --- the engine ------------------------------------------------------------------------------

pub enum Finish {
    Stop,
    Length,
    Cancelled,
}

impl Finish {
    pub fn as_str(&self) -> &'static str {
        match self {
            Finish::Stop => "stop",
            Finish::Length => "length",
            Finish::Cancelled => "cancelled",
        }
    }
}

pub struct Decoded {
    pub tokens: Vec<i64>,
    pub finish: Finish,
    pub prefill_s: f64,
    pub decode_s: f64,
}

pub struct Gemma {
    pub cfg: Config,
    context: i64,
    weights: Vec<Buffer>,
    shapes: Vec<Vec<i64>>,
    caches: Vec<Buffer>,
    prefill: HashMap<i64, Exe>,
    decode: Exe,
    pub weight_bytes: i64,
    pub load_s: f64,
    pub compile_s: f64,
}

impl Gemma {
    pub fn load(dir: &Path, context: i64) -> Result<Gemma> {
        let t0 = Instant::now();
        let text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| format!("{}/config.json: {e}", dir.display()))?;
        let json: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        let cfg = Config::from_json(&json)?;
        let ck = Checkpoint::open(dir)?;
        let mut weights = Vec::new();
        let mut shapes = Vec::new();
        let mut weight_bytes = 0;
        for name in weight_list(&cfg) {
            let (buf, shape) = ck.upload_bf16(&name)?;
            weight_bytes += buf.device_bytes();
            weights.push(buf);
            shapes.push(shape);
        }
        let load_s = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let decode = Exe::compile(&decode_module(&cfg, &shapes, context))?;
        let compile_s = t1.elapsed().as_secs_f64();
        let mut g = Gemma {
            cfg,
            context,
            weights,
            shapes,
            caches: Vec::new(),
            prefill: HashMap::new(),
            decode,
            weight_bytes,
            load_s,
            compile_s,
        };
        g.reset_caches()?;
        Ok(g)
    }

    pub fn context(&self) -> i64 {
        self.context
    }

    fn reset_caches(&mut self) -> Result<()> {
        self.caches.clear();
        for i in 0..self.cfg.first_shared() {
            let shape = [self.cfg.kv_heads, self.context, self.cfg.dh(i)];
            self.caches.push(Buffer::zeros_bf16(&shape)?);
            self.caches.push(Buffer::zeros_bf16(&shape)?);
        }
        Ok(())
    }

    pub fn cache_bytes(&self) -> i64 {
        self.caches.iter().map(Buffer::device_bytes).sum()
    }

    /// Compiles the prefill program for a prompt bucket on first use.
    fn ensure_prefill(&mut self, s: i64) -> Result<()> {
        if !self.prefill.contains_key(&s) {
            let exe = Exe::compile(&prefill_module(&self.cfg, &self.shapes, s, self.context))?;
            self.prefill.insert(s, exe);
        }
        Ok(())
    }

    /// Reads the prompt. Returns the first generated token and the last position's logits.
    pub fn read_prompt(&mut self, ids: &[i64]) -> Result<(i64, Vec<f32>)> {
        let n = ids.len() as i64;
        let s = ((n + BUCKET - 1) / BUCKET) * BUCKET;
        if s > self.context {
            return Err(format!(
                "prompt bucket {s} exceeds context {}",
                self.context
            ));
        }
        let mut padded: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        padded.resize(s as usize, 0);
        let idb = Buffer::vec_i32(&padded)?;
        let lenb = Buffer::scalar_i32(n as i32)?;
        let vocab = self.cfg.vocab as usize;
        self.ensure_prefill(s)?;
        let caches = std::mem::take(&mut self.caches);
        let outs = {
            let exe = &self.prefill[&s];
            let mut args: Vec<&Buffer> = self.weights.iter().collect();
            args.push(&idb);
            args.push(&lenb);
            args.extend(caches.iter());
            exe.run(&args)?
        };
        drop(caches); // donated
        let mut it = outs.into_iter();
        let tok = it.next().unwrap();
        let lg = it.next().unwrap();
        self.caches = it.collect();
        Ok((tok.read_i32()? as i64, lg.read_f32s(vocab)?))
    }

    fn step(&mut self, tok: i64, pos: i64) -> Result<i64> {
        let tb = Buffer::scalar_i32(tok as i32)?;
        let pb = Buffer::scalar_i32(pos as i32)?;
        let caches = std::mem::take(&mut self.caches);
        let outs = {
            let mut args: Vec<&Buffer> = self.weights.iter().collect();
            args.push(&tb);
            args.push(&pb);
            args.extend(caches.iter());
            self.decode.run(&args)?
        };
        drop(caches);
        let mut it = outs.into_iter();
        let next = it.next().unwrap();
        self.caches = it.collect();
        Ok(next.read_i32()? as i64)
    }

    /// Greedy decode: stops at a token in `stop`, after `max_new` tokens, or when `on_tokens`
    /// returns false. Each token is handed to `on_tokens` as it reaches the host.
    pub fn decode(
        &mut self,
        ids: &[i64],
        max_new: i64,
        stop: &[i64],
        mut on_tokens: impl FnMut(&[i64]) -> bool,
    ) -> Result<Decoded> {
        let n = ids.len() as i64;
        if n + max_new > self.context {
            return Err(format!(
                "{n} prompt tokens + {max_new} new tokens exceeds context {}",
                self.context
            ));
        }
        let t0 = Instant::now();
        let (mut tok, _) = self.read_prompt(ids)?;
        let prefill_s = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let mut out = Vec::new();
        let mut finish = Finish::Length;
        for step in 0..max_new {
            if stop.contains(&tok) {
                finish = Finish::Stop;
                break;
            }
            out.push(tok);
            if !on_tokens(&[tok]) {
                finish = Finish::Cancelled;
                break;
            }
            if step == max_new - 1 {
                break;
            }
            tok = self.step(tok, n + step)?;
        }
        Ok(Decoded {
            tokens: out,
            finish,
            prefill_s,
            decode_s: t1.elapsed().as_secs_f64(),
        })
    }
}
