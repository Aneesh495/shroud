//! Rhai execution engine wrapper with safety quotas and execution sandboxing.

use crate::bindings::register_shroud_types;
use rhai::{Engine, EvalAltResult, Scope, AST};
use std::path::{Path, PathBuf};

pub struct ShroudRhaiEngine {
    engine: Engine,
}

impl Default for ShroudRhaiEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct ScriptRunResult {
    pub file_path: String,
    pub passed: bool,
    pub error: Option<String>,
}

impl ShroudRhaiEngine {
    pub fn new() -> Self {
        let mut engine = Engine::new();
        engine.set_max_operations(50_000_000);
        engine.set_max_call_levels(128);
        engine.set_max_expr_depths(64, 64);
        register_shroud_types(&mut engine);
        Self { engine }
    }

    pub fn compile_script(&self, script: &str) -> Result<AST, String> {
        self.engine.compile(script).map_err(|e| format!("{e}"))
    }

    pub fn eval_script<T: Clone + Send + Sync + 'static>(
        &self,
        script: &str,
    ) -> Result<T, Box<EvalAltResult>> {
        let mut scope = Scope::new();
        self.engine.eval_with_scope(&mut scope, script)
    }

    pub fn eval_file<T: Clone + Send + Sync + 'static>(
        &self,
        path: &Path,
    ) -> Result<T, Box<EvalAltResult>> {
        let mut scope = Scope::new();
        self.engine.eval_file_with_scope(&mut scope, path.to_path_buf())
    }

    pub fn run_file(&self, path: &Path) -> Result<(), Box<EvalAltResult>> {
        let mut scope = Scope::new();
        self.engine.run_file_with_scope(&mut scope, path.to_path_buf())
    }

    /// Recursively discovers and executes all `.rhai` scripts under `dir`.
    pub fn run_directory(&self, dir: &Path) -> Vec<ScriptRunResult> {
        let mut files = collect_rhai_files(dir);
        files.sort();

        files
            .into_iter()
            .map(|path| {
                let path_str = path.display().to_string();
                match self.run_file(&path) {
                    Ok(()) => ScriptRunResult {
                        file_path: path_str,
                        passed: true,
                        error: None,
                    },
                    Err(e) => ScriptRunResult {
                        file_path: path_str,
                        passed: false,
                        error: Some(format!("{e}")),
                    },
                }
            })
            .collect()
    }
}

fn collect_rhai_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                out.extend(collect_rhai_files(&p));
            } else if p.extension().and_then(|e| e.to_str()) == Some("rhai") {
                out.push(p);
            }
        }
    }
    out
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
            let c = a + b;
            assert_close(c.x, 5.0, 1e-6);
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

    #[test]
    fn test_rhai_ray_and_scenario_dsl() {
        let engine = ShroudRhaiEngine::new();
        let script = r#"
            let sc = scenario("dense_fog_highway");
            sc.set_fog(0.05, 0.85);
            sc.add_target("sedan", vec3(20.0, -1.0, 0.0), vec3(24.5, 1.0, 1.5), 0.35, 310.0);
            assert_true(sc.target_count() == 1, "target added");
            assert_close(sc.extinction(), 0.05, 1e-6);

            let r = ray(vec3(0.0, 0.0, 0.75), vec3(1.0, 0.0, 0.0));
            let hit_t = r.intersect_box(vec3(20.0, -1.0, 0.0), vec3(24.5, 1.0, 1.5));
            assert_close(hit_t, 20.0, 1e-4);

            let radar = fmcw_config(77.0e9, 1.0e9, 40.0e-6, 0.1);
            let fu = radar.beat_up(85.5, -18.2);
            let fd = radar.beat_down(85.5, -18.2);
            assert_close(radar.solve_range(fu, fd), 85.5, 1e-2);
            assert_close(radar.solve_velocity(fu, fd), -18.2, 1e-2);
            true
        "#;
        let ok: bool = engine.eval_script(script).expect("Scenario eval failed");
        assert!(ok);
    }

    #[test]
    fn test_rhai_assert_failure_returns_error() {
        let engine = ShroudRhaiEngine::new();
        let script = r#"assert_close(1.0, 2.0, 1e-3);"#;
        assert!(engine.eval_script::<()>(script).is_err());
    }
}
