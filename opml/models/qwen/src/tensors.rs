//! Minimal safetensors reader (format: u64 LE header length, JSON header
//! mapping tensor name → {dtype, shape, data_offsets}, then raw data).
//! Hand-rolled: ~60 lines beats a dependency for a consensus-adjacent
//! artifact pipeline.

use serde_json::Value;
use std::collections::BTreeMap;

pub struct SafeTensors {
    /// Raw file bytes (data section offsets are relative to `data_start`).
    bytes: Vec<u8>,
    data_start: usize,
    index: BTreeMap<String, (String, Vec<usize>, usize, usize)>, // dtype, shape, begin, end
}

impl SafeTensors {
    pub fn load(path: &str) -> Self {
        let bytes = std::fs::read(path).expect("read safetensors");
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let header: Value =
            serde_json::from_slice(&bytes[8..8 + header_len]).expect("safetensors header");
        let mut index = BTreeMap::new();
        for (name, meta) in header.as_object().expect("header object") {
            if name == "__metadata__" {
                continue;
            }
            let dtype = meta["dtype"].as_str().expect("dtype").to_string();
            let shape: Vec<usize> = meta["shape"]
                .as_array()
                .expect("shape")
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let off = meta["data_offsets"].as_array().expect("offsets");
            index.insert(
                name.clone(),
                (
                    dtype,
                    shape,
                    off[0].as_u64().unwrap() as usize,
                    off[1].as_u64().unwrap() as usize,
                ),
            );
        }
        Self { bytes, data_start: 8 + header_len, index }
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.index.keys()
    }

    pub fn shape(&self, name: &str) -> &[usize] {
        &self.index.get(name).unwrap_or_else(|| panic!("tensor {name}")).1
    }

    /// Raw bf16 element bits (LE u16 pairs) for a tensor.
    pub fn bf16_bits(&self, name: &str) -> Vec<u16> {
        let (dtype, _, b, e) = self.index.get(name).unwrap_or_else(|| panic!("tensor {name}"));
        assert_eq!(dtype, "BF16", "{name}: expected BF16, got {dtype}");
        self.bytes[self.data_start + b..self.data_start + e]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }
}
