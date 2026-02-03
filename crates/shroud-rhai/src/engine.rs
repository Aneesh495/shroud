//! Rhai execution engine wrapper with safety quotas and execution sandboxing.

use crate::bindings::register_shroud_types;
use rhai::{Engine, EvalAltResult, Scope, AST};
use std::path::Path;

pub struct ShroudRhaiEngine {
    engine: Engine,
}

impl Default for ShroudRhaiEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ShroudRhaiEngine {
    pub fn new() -> Self {
        let mut engine = Engine::new();
        engine.set_max_operations(10_000_000);
        engine.set_max_call_levels(128);
        register_shroud_types(&mut engine);
        Self { engine }
    }

    pub fn compile_script(&self, script: &str) -> Result<AST, String> {
        self.engine.compile(script).map_err(|e| format!("{e}"))
    }

    pub fn eval_script<T: Clone + Send + Sync + 'static>(&self, script: &str) -> Result<T, Box<EvalAltResult>> {
        let mut scope = Scope::new();
        self.engine.eval_with_scope(&mut scope, script)
    }

    pub fn eval_file<T: Clone + Send + Sync + 'static>(&self, path: &Path) -> Result<T, Box<EvalAltResult>> {
        let mut scope = Scope::new();
        self.engine.eval_file_with_scope(&mut scope, path.to_path_buf())
    }

    pub fn run_file(&self, path: &Path) -> Result<(), Box<EvalAltResult>> {
        let mut scope = Scope::new();
        self.engine.run_file_with_scope(&mut scope, path.to_path_buf())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rhai_vec3_evaluation() {
        let engine = ShroudRhaiEngine::new();
        let script = r#"
            let a = vec3(1.0, 2.0, 3.0);
            let b = vec3(4.0, 5.0, 6.0);
            a.dot(b)
        "#;
        let res: f64 = engine.eval_script(script).expect("Eval failed");
        assert_eq!(res, 32.0);
    }

    #[test]
    fn test_rhai_henyey_greenstein_binding() {
        let engine = ShroudRhaiEngine::new();
        let script = r#"
            let p_fwd = eval_henyey_greenstein(1.0, 0.5);
            let p_back = eval_henyey_greenstein(-1.0, 0.5);
            p_fwd > p_back
        "#;
        let res: bool = engine.eval_script(script).expect("Eval failed");
        assert!(res);
    }
}
