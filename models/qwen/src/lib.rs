//! Qwen3 integer port (Phase 3).
//!
//! Float policy (SPEC §6.3): floats appear ONLY in `quant` (the offline
//! quantizer/calibrator, loudly opted out of the workspace float ban).
//! Everything the runtime touches — including rotary sin/cos tables — is
//! generated with pure integer arithmetic (`trig`), so golden hashes can
//! never drift across platforms via libm.

pub mod config;
pub mod fmath;
pub mod fmodel;
pub mod forward;
pub mod image;
pub mod layout;
pub mod quant;
pub mod tensors;
pub mod trig;
