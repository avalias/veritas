//! Shared toy-judge setup: model + compiled program + genesis image,
//! built once and reused by every actor and test (it is all deterministic).

use compiler::{compile_toy, Compiled};
use toy_model::layout::{genesis_image, Layout, MEM_DEPTH};
use toy_model::model::{tokenize, ToyModel, WEIGHT_SEED};
use vm::exec::Machine;
use vm::hash::Hash;
use vm::onestep::{JudgeParams, ProgramTree};

pub struct ToySetup {
    pub lay: Layout,
    pub compiled: Compiled,
    pub image: Vec<u8>,
    pub program_tree: ProgramTree,
    pub judge: JudgeParams,
    pub genesis_root: Hash,
}

impl ToySetup {
    pub fn new(prompt: &str, n_gen: usize) -> Self {
        let lay = Layout::new();
        let model = ToyModel::generate(WEIGHT_SEED);
        let toks = tokenize(prompt);
        let compiled = compile_toy(&lay, toks.len(), n_gen);
        let image = genesis_image(&lay, &model, &toks);
        let program_tree = ProgramTree::new(&compiled.program, compiled.p);
        let judge = JudgeParams {
            d: MEM_DEPTH,
            p: compiled.p,
            program_root: program_tree.root(),
        };
        let genesis_root = Self::machine_from(&compiled, &image).state_root();
        Self { lay, compiled, image, program_tree, judge, genesis_root }
    }

    fn machine_from(compiled: &Compiled, image: &[u8]) -> Machine {
        Machine::with_image(MEM_DEPTH, compiled.p, compiled.program.clone(), image)
    }

    /// Fresh machine at genesis.
    pub fn machine(&self) -> Machine {
        Self::machine_from(&self.compiled, &self.image)
    }
}
