//! Reads a Hugging Face safetensors checkpoint (one file or an index of shards) through mmap
//! and hands back raw tensor bytes, one tensor at a time, so host memory stays at one tensor.
//! No Python here.
use crate::pjrt::{Buffer, Elem, Result};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

pub struct Checkpoint {
    maps: Vec<Mmap>,
    /// tensor name -> index into `maps`
    index: HashMap<String, usize>,
}

impl Checkpoint {
    pub fn open(dir: &Path) -> Result<Checkpoint> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(format!("no .safetensors in {}", dir.display()));
        }
        let mut maps = Vec::new();
        let mut index = HashMap::new();
        for f in &files {
            let file = File::open(f).map_err(|e| format!("{}: {e}", f.display()))?;
            let map = unsafe { Mmap::map(&file) }.map_err(|e| format!("{}: {e}", f.display()))?;
            let names: Vec<String> = SafeTensors::deserialize(&map)
                .map_err(|e| format!("{}: {e}", f.display()))?
                .names()
                .into_iter()
                .map(|n| n.to_string())
                .collect();
            for n in names {
                index.insert(n, maps.len());
            }
            maps.push(map);
        }
        Ok(Checkpoint { maps, index })
    }

    /// Shape of a tensor, without reading its data.
    pub fn shape(&self, name: &str) -> Result<Vec<i64>> {
        let (_, shape, _) = self.raw(name)?;
        Ok(shape)
    }

    fn raw(&self, name: &str) -> Result<(Dtype, Vec<i64>, &[u8])> {
        let i = *self
            .index
            .get(name)
            .ok_or_else(|| format!("checkpoint has no tensor {name}"))?;
        let st = SafeTensors::deserialize(&self.maps[i]).map_err(|e| e.to_string())?;
        let t = st.tensor(name).map_err(|e| e.to_string())?;
        let shape = t.shape().iter().map(|&d| d as i64).collect();
        // The view borrows the mmap, which lives as long as self.
        let data: &[u8] = unsafe { std::slice::from_raw_parts(t.data().as_ptr(), t.data().len()) };
        Ok((t.dtype(), shape, data))
    }

    /// Uploads a bf16 tensor to the device as stored. Returns the buffer and its shape.
    pub fn upload_bf16(&self, name: &str) -> Result<(Buffer, Vec<i64>)> {
        let (dt, shape, data) = self.raw(name)?;
        if dt != Dtype::BF16 {
            return Err(format!(
                "{name} is {dt:?}; this engine loads bf16 checkpoints"
            ));
        }
        Ok((Buffer::upload(Elem::BF16, data, &shape)?, shape))
    }
}
