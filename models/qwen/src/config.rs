//! HF config.json reader — values are READ, never guessed (brief Phase 3).

use serde_json::Value;

#[derive(Clone, Debug)]
pub struct QwenConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rope_theta: u64,
    pub rms_norm_eps_recip: u64, // 1/eps as integer (eps = 1e-6 ⇒ 1_000_000)
    pub eos_token_id: u32,
    pub tie_word_embeddings: bool,
}

impl QwenConfig {
    pub fn load(path: &str) -> Self {
        let v: Value =
            serde_json::from_str(&std::fs::read_to_string(path).expect("read config.json"))
                .expect("parse config.json");
        let u = |k: &str| -> usize {
            v[k].as_u64().unwrap_or_else(|| panic!("config field {k}")) as usize
        };
        assert_eq!(v["model_type"].as_str(), Some("qwen3"), "expected a qwen3 config");
        assert_eq!(v["hidden_act"].as_str(), Some("silu"), "expected SiLU MLP");
        assert_eq!(v["attention_bias"].as_bool(), Some(false), "biasless attention assumed");
        assert!(v["rope_scaling"].is_null(), "rope scaling not implemented");
        // eps comes as a float literal in JSON; we only support the exact
        // value 1e-6 (asserted) and carry its reciprocal as an integer.
        let eps = v["rms_norm_eps"].as_f64().expect("rms_norm_eps");
        // Bit-equality (no float arithmetic in this crate's integer side).
        assert!(eps.to_bits() == 1e-6f64.to_bits(), "only eps = 1e-6 supported, got {eps}");
        Self {
            hidden_size: u("hidden_size"),
            num_hidden_layers: u("num_hidden_layers"),
            num_attention_heads: u("num_attention_heads"),
            num_key_value_heads: u("num_key_value_heads"),
            head_dim: u("head_dim"),
            intermediate_size: u("intermediate_size"),
            vocab_size: u("vocab_size"),
            rope_theta: v["rope_theta"].as_u64().expect("rope_theta"),
            rms_norm_eps_recip: 1_000_000,
            eos_token_id: v["eos_token_id"].as_u64().expect("eos_token_id") as u32,
            tie_word_embeddings: v["tie_word_embeddings"].as_bool().unwrap_or(false),
        }
    }
}
