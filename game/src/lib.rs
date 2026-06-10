//! The fraud game (SPEC §8): traces, fault injection, actors, MockChain.
//!
//! MockChain enforces the protocol with logical-tick time; the Sui Move
//! package (Phase 2) replaces it with `sui::clock` and real bonds, reusing
//! the same `vm::onestep::verify_step` semantics.

pub mod actors;
pub mod chain;
pub mod driver;
pub mod setup;
pub mod trace;
