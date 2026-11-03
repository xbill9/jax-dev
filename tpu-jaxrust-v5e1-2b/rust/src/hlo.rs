//! A StableHLO emitter: builds one `func.func @main` as MLIR text, which PJRT compiles.
//!
//! Every op is written in MLIR's generic form (`"stablehlo.add"(%a, %b) : (...) -> ...`), so
//! the text depends only on op names and attribute syntax, not on each op's pretty-printer.
//! Values carry their shape and element type, and the binary ops promote bf16 to f32 the way
//! JAX does, so a graph written here has the element types the JAX engine's graph has.
use std::fmt::Write;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    BF16,
    S32,
    Pred,
}

impl DType {
    pub fn mlir(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::BF16 => "bf16",
            DType::S32 => "i32",
            DType::Pred => "i1",
        }
    }
    pub fn is_float(self) -> bool {
        matches!(self, DType::F32 | DType::BF16)
    }
}

/// An SSA value: its name, shape and element type.
#[derive(Clone, Debug)]
pub struct V {
    name: String,
    pub shape: Vec<i64>,
    pub dt: DType,
}

impl V {
    pub fn rank(&self) -> usize {
        self.shape.len()
    }
    pub fn dim(&self, i: isize) -> i64 {
        let i = if i < 0 {
            self.shape.len() as isize + i
        } else {
            i
        } as usize;
        self.shape[i]
    }
}

pub fn ty(shape: &[i64], dt: DType) -> String {
    if shape.is_empty() {
        format!("tensor<{}>", dt.mlir())
    } else {
        let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        format!("tensor<{}x{}>", dims.join("x"), dt.mlir())
    }
}

fn vty(v: &V) -> String {
    ty(&v.shape, v.dt)
}

fn arr(xs: &[i64]) -> String {
    if xs.is_empty() {
        return "array<i64>".to_string();
    }
    let s: Vec<String> = xs.iter().map(|d| d.to_string()).collect();
    format!("array<i64: {}>", s.join(", "))
}

fn list(xs: &[i64]) -> String {
    let s: Vec<String> = xs.iter().map(|d| d.to_string()).collect();
    format!("[{}]", s.join(", "))
}

/// A float literal MLIR accepts for any float type.
fn flit(x: f64) -> String {
    if x.is_infinite() {
        // Hex form: the bit pattern of -inf / +inf in f32 and bf16 alike is not shared, so
        // large finite stand-ins are used instead (see `Builder::splat`).
        unreachable!("infinite literal")
    }
    // MLIR's float literal needs a '.': `{:e}` writes `2e0`, which lexes as the integer 2.
    let s = format!("{x:e}");
    match s.find('e') {
        Some(i) if !s[..i].contains('.') => format!("{}.0{}", &s[..i], &s[i..]),
        _ => s,
    }
}

#[derive(Clone, Copy)]
pub enum Cmp {
    Lt,
    Le,
    Gt,
}

impl Cmp {
    fn s(self) -> &'static str {
        match self {
            Cmp::Lt => "LT",
            Cmp::Le => "LE",
            Cmp::Gt => "GT",
        }
    }
}

pub struct Builder {
    body: String,
    next: usize,
    params: Vec<(V, Option<usize>)>,
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

impl Builder {
    pub fn new() -> Self {
        Builder {
            body: String::new(),
            next: 0,
            params: Vec::new(),
        }
    }

    fn fresh(&mut self) -> String {
        self.next += 1;
        format!("%v{}", self.next)
    }

    /// A function argument. `alias` names the result it is donated to (`tf.aliasing_output`),
    /// which lets the KV cache be updated in place.
    pub fn param(&mut self, shape: &[i64], dt: DType, alias: Option<usize>) -> V {
        let v = V {
            name: format!("%a{}", self.params.len()),
            shape: shape.to_vec(),
            dt,
        };
        self.params.push((v.clone(), alias));
        v
    }

    fn op(&mut self, name: &str, ins: &[&V], attrs: &str, shape: &[i64], dt: DType) -> V {
        let out = self.fresh();
        let names: Vec<&str> = ins.iter().map(|v| v.name.as_str()).collect();
        let tys: Vec<String> = ins.iter().map(|v| vty(v)).collect();
        let attrs = if attrs.is_empty() {
            String::new()
        } else {
            format!(" {{{attrs}}}")
        };
        writeln!(
            self.body,
            "    {out} = \"stablehlo.{name}\"({}){attrs} : ({}) -> {}",
            names.join(", "),
            tys.join(", "),
            ty(shape, dt)
        )
        .unwrap();
        V {
            name: out,
            shape: shape.to_vec(),
            dt,
        }
    }

    // --- constants -----------------------------------------------------------------------

    /// A tensor of `shape` with every element `x`.
    pub fn splat(&mut self, x: f64, shape: &[i64], dt: DType) -> V {
        let lit = match dt {
            DType::Pred => (if x != 0.0 { "true" } else { "false" }).to_string(),
            DType::S32 => format!("{}", x as i64),
            _ => flit(x),
        };
        let out = self.fresh();
        writeln!(
            self.body,
            "    {out} = \"stablehlo.constant\"() {{value = dense<{lit}> : {t}}} : () -> {t}",
            t = ty(shape, dt)
        )
        .unwrap();
        V {
            name: out,
            shape: shape.to_vec(),
            dt,
        }
    }

    pub fn scalar(&mut self, x: f64, dt: DType) -> V {
        self.splat(x, &[], dt)
    }

    /// A 1-D f32 constant with the given values.
    pub fn f32_vec(&mut self, xs: &[f32]) -> V {
        let vals: Vec<String> = xs.iter().map(|x| flit(*x as f64)).collect();
        let shape = [xs.len() as i64];
        let out = self.fresh();
        writeln!(
            self.body,
            "    {out} = \"stablehlo.constant\"() {{value = dense<[{}]> : {t}}} : () -> {t}",
            vals.join(", "),
            t = ty(&shape, DType::F32)
        )
        .unwrap();
        V {
            name: out,
            shape: shape.to_vec(),
            dt: DType::F32,
        }
    }

    pub fn iota(&mut self, shape: &[i64], dim: usize, dt: DType) -> V {
        self.op(
            "iota",
            &[],
            &format!("iota_dimension = {dim} : i64"),
            shape,
            dt,
        )
    }

    // --- shape ---------------------------------------------------------------------------

    pub fn convert(&mut self, v: &V, dt: DType) -> V {
        if v.dt == dt {
            return v.clone();
        }
        self.op("convert", &[v], "", &v.shape.clone(), dt)
    }

    pub fn reshape(&mut self, v: &V, shape: &[i64]) -> V {
        let n: i64 = v.shape.iter().product();
        let m: i64 = shape.iter().product();
        assert_eq!(n, m, "reshape {:?} -> {:?}", v.shape, shape);
        if v.shape == shape {
            return v.clone();
        }
        self.op("reshape", &[v], "", shape, v.dt)
    }

    pub fn transpose(&mut self, v: &V, perm: &[i64]) -> V {
        let shape: Vec<i64> = perm.iter().map(|&p| v.shape[p as usize]).collect();
        self.op(
            "transpose",
            &[v],
            &format!("permutation = {}", arr(perm)),
            &shape,
            v.dt,
        )
    }

    /// Broadcasts `v` to `shape`; `dims[i]` is the output dimension of input dimension i.
    pub fn broadcast(&mut self, v: &V, shape: &[i64], dims: &[i64]) -> V {
        assert_eq!(dims.len(), v.rank());
        if v.shape == shape {
            return v.clone();
        }
        self.op(
            "broadcast_in_dim",
            &[v],
            &format!("broadcast_dimensions = {}", arr(dims)),
            shape,
            v.dt,
        )
    }

    /// Numpy-style trailing broadcast of `v` to `shape`, inserting size-1 matches.
    pub fn bcast_to(&mut self, v: &V, shape: &[i64]) -> V {
        if v.shape == shape {
            return v.clone();
        }
        let off = shape.len() - v.rank();
        // Drop size-1 dims that need to grow, since broadcast_in_dim requires each input
        // dim to equal its output dim or be 1 — which it allows, so keep them.
        let dims: Vec<i64> = (0..v.rank()).map(|i| (off + i) as i64).collect();
        for (i, &d) in v.shape.iter().enumerate() {
            assert!(
                d == 1 || d == shape[off + i],
                "bcast {:?} -> {:?}",
                v.shape,
                shape
            );
        }
        self.broadcast(v, shape, &dims)
    }

    pub fn slice(&mut self, v: &V, start: &[i64], limit: &[i64]) -> V {
        let shape: Vec<i64> = start.iter().zip(limit).map(|(s, l)| l - s).collect();
        let strides = vec![1i64; start.len()];
        self.op(
            "slice",
            &[v],
            &format!(
                "start_indices = {}, limit_indices = {}, strides = {}",
                arr(start),
                arr(limit),
                arr(&strides)
            ),
            &shape,
            v.dt,
        )
    }

    pub fn concat(&mut self, vs: &[&V], dim: usize) -> V {
        let mut shape = vs[0].shape.clone();
        shape[dim] = vs.iter().map(|v| v.shape[dim]).sum();
        self.op(
            "concatenate",
            vs,
            &format!("dimension = {dim} : i64"),
            &shape,
            vs[0].dt,
        )
    }

    /// Writes `upd` into `op` at the scalar s32 start indices `idx`.
    pub fn dynamic_update_slice(&mut self, op: &V, upd: &V, idx: &[&V]) -> V {
        let upd = self.convert(upd, op.dt);
        let mut ins: Vec<&V> = vec![op, &upd];
        ins.extend_from_slice(idx);
        let shape = op.shape.clone();
        self.op("dynamic_update_slice", &ins, "", &shape, op.dt)
    }

    /// The `sizes` block of `op` starting at the scalar s32 indices `idx`.
    pub fn dynamic_slice(&mut self, op: &V, idx: &[&V], sizes: &[i64]) -> V {
        let mut ins: Vec<&V> = vec![op];
        ins.extend_from_slice(idx);
        self.op(
            "dynamic_slice",
            &ins,
            &format!("slice_sizes = {}", arr(sizes)),
            sizes,
            op.dt,
        )
    }

    /// Rows `idx` (s32, shape [S]) of the 2-D `table`: shape [S, table.dim(1)].
    pub fn gather_rows(&mut self, table: &V, idx: &V) -> V {
        let s = idx.shape[0];
        let d = table.shape[1];
        let idx2 = self.reshape(idx, &[s, 1]);
        self.op(
            "gather",
            &[table, &idx2],
            &format!(
                "dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = {}, indices_are_sorted = false",
                arr(&[1, d])
            ),
            &[s, d],
            table.dt,
        )
    }

    // --- elementwise ---------------------------------------------------------------------

    /// JAX's promotion for the two float types: bf16 meets f32 as f32.
    fn promote(&mut self, a: &V, b: &V) -> (V, V) {
        if a.dt == b.dt {
            return (a.clone(), b.clone());
        }
        assert!(
            a.dt.is_float() && b.dt.is_float(),
            "{:?} vs {:?}",
            a.dt,
            b.dt
        );
        (self.convert(a, DType::F32), self.convert(b, DType::F32))
    }

    fn binary(&mut self, name: &str, a: &V, b: &V) -> V {
        let (a, b) = self.promote(a, b);
        let shape = if a.rank() >= b.rank() {
            a.shape.clone()
        } else {
            b.shape.clone()
        };
        let a = self.bcast_to(&a, &shape);
        let b = self.bcast_to(&b, &shape);
        self.op(name, &[&a, &b], "", &shape, a.dt)
    }

    pub fn add(&mut self, a: &V, b: &V) -> V {
        self.binary("add", a, b)
    }
    pub fn sub(&mut self, a: &V, b: &V) -> V {
        self.binary("subtract", a, b)
    }
    pub fn mul(&mut self, a: &V, b: &V) -> V {
        self.binary("multiply", a, b)
    }
    pub fn div(&mut self, a: &V, b: &V) -> V {
        self.binary("divide", a, b)
    }

    /// `v * x` for a Python-float `x`: JAX treats it as weakly typed, so `v` keeps its type.
    pub fn scale(&mut self, v: &V, x: f64) -> V {
        let c = self.splat(x, &v.shape.clone(), v.dt);
        self.op("multiply", &[v, &c], "", &v.shape.clone(), v.dt)
    }

    /// `v + x` for a Python-float `x`, keeping `v`'s type.
    pub fn add_scalar(&mut self, v: &V, x: f64) -> V {
        let c = self.splat(x, &v.shape.clone(), v.dt);
        self.op("add", &[v, &c], "", &v.shape.clone(), v.dt)
    }

    fn unary(&mut self, name: &str, v: &V) -> V {
        self.op(name, &[v], "", &v.shape.clone(), v.dt)
    }
    pub fn rsqrt(&mut self, v: &V) -> V {
        self.unary("rsqrt", v)
    }
    pub fn tanh(&mut self, v: &V) -> V {
        self.unary("tanh", v)
    }
    pub fn exp(&mut self, v: &V) -> V {
        self.unary("exponential", v)
    }
    pub fn cos(&mut self, v: &V) -> V {
        self.unary("cosine", v)
    }
    pub fn sin(&mut self, v: &V) -> V {
        self.unary("sine", v)
    }
    pub fn neg(&mut self, v: &V) -> V {
        self.unary("negate", v)
    }

    pub fn compare(&mut self, a: &V, b: &V, c: Cmp) -> V {
        let shape = if a.rank() >= b.rank() {
            a.shape.clone()
        } else {
            b.shape.clone()
        };
        let a = self.bcast_to(a, &shape);
        let b = self.bcast_to(b, &shape);
        let kind = if a.dt.is_float() { "FLOAT" } else { "SIGNED" };
        self.op(
            "compare",
            &[&a, &b],
            &format!(
                "comparison_direction = #stablehlo<comparison_direction {}>, compare_type = #stablehlo<comparison_type {kind}>",
                c.s()
            ),
            &shape,
            DType::Pred,
        )
    }

    pub fn and(&mut self, a: &V, b: &V) -> V {
        self.op("and", &[a, b], "", &a.shape.clone(), DType::Pred)
    }

    pub fn select(&mut self, p: &V, a: &V, b: &V) -> V {
        let p = self.bcast_to(p, &a.shape.clone());
        self.op("select", &[&p, a, b], "", &a.shape.clone(), a.dt)
    }

    // --- reductions ----------------------------------------------------------------------

    fn reduce(&mut self, v: &V, dims: &[i64], combiner: &str, init: f64) -> V {
        let init = self.scalar(init, v.dt);
        let shape: Vec<i64> = v
            .shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !dims.contains(&(*i as i64)))
            .map(|(_, d)| *d)
            .collect();
        let (x, y, r) = (self.fresh(), self.fresh(), self.fresh());
        let st = ty(&[], v.dt);
        let out = self.fresh();
        writeln!(
            self.body,
            "    {out} = \"stablehlo.reduce\"({}, {}) ({{\n    ^bb0({x}: {st}, {y}: {st}):\n      {r} = \"stablehlo.{combiner}\"({x}, {y}) : ({st}, {st}) -> {st}\n      \"stablehlo.return\"({r}) : ({st}) -> ()\n    }}) {{dimensions = {}}} : ({}, {st}) -> {}",
            v.name,
            init.name,
            arr(dims),
            vty(v),
            ty(&shape, v.dt)
        )
        .unwrap();
        V {
            name: out,
            shape,
            dt: v.dt,
        }
    }

    pub fn reduce_sum(&mut self, v: &V, dims: &[i64]) -> V {
        self.reduce(v, dims, "add", 0.0)
    }

    pub fn reduce_max(&mut self, v: &V, dims: &[i64]) -> V {
        // The most negative finite value: a reduction never sees an empty set here.
        let lo = match v.dt {
            DType::BF16 => -3.3895313892515355e38,
            _ => -3.4028234663852886e38,
        };
        self.reduce(v, dims, "maximum", lo)
    }

    /// Index of the largest element of a 1-D f32 `v`, lowest index on ties (jnp.argmax).
    pub fn argmax(&mut self, v: &V) -> V {
        assert_eq!(v.rank(), 1);
        assert_eq!(v.dt, DType::F32);
        let n = v.shape[0];
        let idx = self.iota(&[n], 0, DType::S32);
        let lo = self.scalar(-3.4028234663852886e38, DType::F32);
        let zero = self.scalar(0.0, DType::S32);
        let names: Vec<String> = (0..10).map(|_| self.fresh()).collect();
        let [a, ai, b, bi, gt, eq, lt, take, vo, io] = [
            &names[0], &names[1], &names[2], &names[3], &names[4], &names[5], &names[6], &names[7],
            &names[8], &names[9],
        ];
        let f = ty(&[], DType::F32);
        let s = ty(&[], DType::S32);
        let p = ty(&[], DType::Pred);
        let out = self.fresh();
        let outi = format!("{out}#1");
        writeln!(
            self.body,
            "    {out}:2 = \"stablehlo.reduce\"({}, {}, {}, {}) ({{\n    ^bb0({a}: {f}, {ai}: {s}, {b}: {f}, {bi}: {s}):\n      {gt} = \"stablehlo.compare\"({a}, {b}) {{comparison_direction = #stablehlo<comparison_direction GT>, compare_type = #stablehlo<comparison_type FLOAT>}} : ({f}, {f}) -> {p}\n      {eq} = \"stablehlo.compare\"({a}, {b}) {{comparison_direction = #stablehlo<comparison_direction EQ>, compare_type = #stablehlo<comparison_type FLOAT>}} : ({f}, {f}) -> {p}\n      {lt} = \"stablehlo.compare\"({ai}, {bi}) {{comparison_direction = #stablehlo<comparison_direction LT>, compare_type = #stablehlo<comparison_type SIGNED>}} : ({s}, {s}) -> {p}\n      {take} = \"stablehlo.and\"({eq}, {lt}) : ({p}, {p}) -> {p}\n      {take}o = \"stablehlo.or\"({gt}, {take}) : ({p}, {p}) -> {p}\n      {vo} = \"stablehlo.select\"({take}o, {a}, {b}) : ({p}, {f}, {f}) -> {f}\n      {io} = \"stablehlo.select\"({take}o, {ai}, {bi}) : ({p}, {s}, {s}) -> {s}\n      \"stablehlo.return\"({vo}, {io}) : ({f}, {s}) -> ()\n    }}) {{dimensions = array<i64: 0>}} : ({}, {}, {f}, {s}) -> ({f}, {s})",
            v.name,
            idx.name,
            lo.name,
            zero.name,
            vty(v),
            vty(&idx)
        )
        .unwrap();
        V {
            name: outi,
            shape: vec![],
            dt: DType::S32,
        }
    }

    // --- contractions --------------------------------------------------------------------

    /// dot_general with JAX's promotion: bf16 x bf16 stays bf16, anything with f32 is f32.
    /// Output dims: batch, then lhs free, then rhs free.
    pub fn dot(&mut self, a: &V, b: &V, lb: &[i64], rb: &[i64], lc: &[i64], rc: &[i64]) -> V {
        let (a, b) = self.promote(a, b);
        let mut shape: Vec<i64> = lb.iter().map(|&d| a.shape[d as usize]).collect();
        for (i, &d) in a.shape.iter().enumerate() {
            let i = i as i64;
            if !lb.contains(&i) && !lc.contains(&i) {
                shape.push(d);
            }
        }
        for (i, &d) in b.shape.iter().enumerate() {
            let i = i as i64;
            if !rb.contains(&i) && !rc.contains(&i) {
                shape.push(d);
            }
        }
        for (x, y) in lc.iter().zip(rc) {
            assert_eq!(
                a.shape[*x as usize], b.shape[*y as usize],
                "contracting dims"
            );
        }
        self.op(
            "dot_general",
            &[&a, &b],
            &format!(
                "dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = {}, rhs_batching_dimensions = {}, lhs_contracting_dimensions = {}, rhs_contracting_dimensions = {}>",
                list(lb),
                list(rb),
                list(lc),
                list(rc)
            ),
            &shape,
            a.dt,
        )
    }

    /// `x @ w.T` for a checkpoint weight stored `[out, in]`: contracts x's last dim with w's
    /// dim 1, the layout the JAX engine reaches by transposing at load.
    pub fn linear(&mut self, x: &V, w: &V) -> V {
        let r = x.rank() as i64;
        self.dot(x, w, &[], &[], &[r - 1], &[1])
    }

    // --- module --------------------------------------------------------------------------

    /// The finished module, returning `results` in order.
    pub fn finish(self, results: &[&V]) -> String {
        let mut args = Vec::new();
        for (v, alias) in &self.params {
            let a = match alias {
                Some(i) => format!(" {{tf.aliasing_output = {i} : i32}}"),
                None => String::new(),
            };
            args.push(format!("{}: {}{a}", v.name, vty(v)));
        }
        let rtys: Vec<String> = results.iter().map(|v| vty(v)).collect();
        let rnames: Vec<&str> = results.iter().map(|v| v.name.as_str()).collect();
        format!(
            "module @jaxrust attributes {{mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32}} {{\n  func.func public @main({}) -> ({}) {{\n{}    \"func.return\"({}) : ({}) -> ()\n  }}\n}}\n",
            args.join(", "),
            rtys.join(", "),
            self.body,
            rnames.join(", "),
            rtys.join(", ")
        )
    }
}
