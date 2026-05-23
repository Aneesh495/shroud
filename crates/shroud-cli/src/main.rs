//! Command-line interface for Shroud: validation, headless GPU execution, benchmarks, and Rhai testing.

use std::env;
use std::path::Path;
use std::process::exit;
use std::time::Instant;

fn print_usage() {
    println!("Shroud: WebGPU Spectral Radiative Transfer & Multi-Sensor Simulation Engine");
    println!("Usage:");
    println!("  shroud validate <path-to-shader-or-dir>");
    println!("  shroud rhai <path-to-script>");
    println!("  shroud test-rhai <path-to-script-or-dir>");
    println!("  shroud bench [shaders-dir]");
    println!("  shroud verify");
    println!("  shroud info");
}

fn validate_path(path: &Path) -> bool {
    if path.is_dir() {
        println!("Validating WGSL shader directory: {}...", path.display());
        let start = Instant::now();
        let results = shroud_validator::validate_wgsl_directory(path);
        let elapsed = start.elapsed();
        let mut passed = 0usize;
        let mut failed = 0usize;
        for res in &results {
            if res.is_valid {
                passed += 1;
            } else {
                failed += 1;
                eprintln!("FAIL: {}\n{}", res.file_path, res.errors.join("\n"));
            }
        }
        println!(
            "Validation summary: {} passed, {} failed out of {} WGSL shaders in {:.2?}.",
            passed,
            failed,
            results.len(),
            elapsed
        );
        if failed == 0 {
            println!("PASS: All WGSL shaders compiled and validated successfully via Naga.");
            true
        } else {
            false
        }
    } else if path.is_file() {
        println!("Validating WGSL shader: {}...", path.display());
        match std::fs::read_to_string(path) {
            Ok(source) => {
                let res =
                    shroud_validator::validate_wgsl_source(&path.display().to_string(), &source);
                if res.is_valid {
                    println!(
                        "PASS: WGSL shader validated successfully. Entry points: {:?}",
                        res.entry_points
                    );
                    true
                } else {
                    eprintln!("FAIL: WGSL validation failed:\n{}", res.errors.join("\n"));
                    false
                }
            }
            Err(e) => {
                eprintln!("Error reading file: {}", e);
                false
            }
        }
    } else {
        eprintln!("Error: path does not exist: {}", path.display());
        false
    }
}

fn test_rhai_path(path: &Path) -> bool {
    let runner = shroud_rhai::ShroudRhaiEngine::new();
    if path.is_dir() {
        println!("Executing Rhai scenario suite under: {}...", path.display());
        let start = Instant::now();
        let results = runner.run_directory(path);
        let elapsed = start.elapsed();
        let mut passed = 0usize;
        let mut failed = 0usize;
        for res in &results {
            if res.passed {
                passed += 1;
            } else {
                failed += 1;
                eprintln!(
                    "FAIL: {}\n  {}",
                    res.file_path,
                    res.error.as_deref().unwrap_or("unknown error")
                );
            }
        }
        println!(
            "Rhai suite summary: {} passed, {} failed out of {} scripts in {:.2?}.",
            passed,
            failed,
            results.len(),
            elapsed
        );
        failed == 0
    } else if path.is_file() {
        println!("Executing Rhai script: {}...", path.display());
        match runner.run_file(path) {
            Ok(()) => {
                println!("PASS: Rhai script executed successfully.");
                true
            }
            Err(e) => {
                eprintln!("FAIL: Script execution error: {}", e);
                false
            }
        }
    } else {
        eprintln!("Error: path does not exist: {}", path.display());
        false
    }
}

fn run_bench(shaders_dir: &Path) {
    println!("==================================================================");
    println!("  SHROUD PERFORMANCE & COMPUTE PIPELINE BENCHMARK");
    println!("==================================================================");

    let t0 = Instant::now();
    let results = shroud_validator::validate_wgsl_directory(shaders_dir);
    let naga_dt = t0.elapsed();
    let valid_count = results.iter().filter(|r| r.is_valid).count();
    println!(
        "[Naga AST + Type Validation] {}/{} WGSL modules in {:.2?}",
        valid_count,
        results.len(),
        naga_dt
    );

    let rhai_start = Instant::now();
    let engine = shroud_rhai::ShroudRhaiEngine::new();
    let rhai_bench = r#"
        let acc = 0.0;
        let i = 0;
        while i < 5000 {
            let ct = -1.0 + 2.0 * (i / 5000.0);
            acc += eval_henyey_greenstein(ct, 0.72);
            i += 1;
        }
        acc
    "#;
    let _sum: f64 = engine.eval_script(rhai_bench).unwrap_or(0.0);
    println!(
        "[Rhai Physics DSL] 5,000 Henyey-Greenstein phase evaluations in {:.2?}",
        rhai_start.elapsed()
    );

    // Attempt headless WGPU compute dispatch on available GPU adapter
    match pollster::block_on(shroud_wgpu::context::GpuContext::init_headless()) {
        Ok(ctx) => {
            let info = ctx.adapter.get_info();
            println!("[WGPU Adapter] {} ({:?})", info.name, info.backend);
            let test_wgsl = r#"
                @group(0) @binding(0) var<storage, read_write> buf: array<f32>;
                @compute @workgroup_size(64, 1, 1)
                fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
                    let i = gid.x;
                    if (i < arrayLength(&buf)) {
                        buf[i] = sqrt(f32(i) + 1.0);
                    }
                }
            "#;
            let t_pipe = Instant::now();
            let count = 65_536usize;
            match pollster::block_on(
                shroud_wgpu::pipeline::ComputePipelineHelper::dispatch_storage_f32(
                    &ctx.device,
                    &ctx.queue,
                    "shroud_bench_kernel",
                    test_wgsl,
                    "cs_main",
                    count,
                ),
            ) {
                Ok(vals) => {
                    println!(
                        "[WGPU Headless Dispatch] {} work-items computed & read back in {:.2?} (sample[99]={:.3})",
                        vals.len(),
                        t_pipe.elapsed(),
                        vals[99]
                    );
                }
                Err(e) => println!("[WGPU Dispatch Error] {}", e),
            }
        }
        Err(e) => {
            println!("[WGPU Headless Adapter Unavailable] {}", e);
        }
    }
    println!("==================================================================");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        exit(0);
    }

    match args[1].as_str() {
        "validate" => {
            if args.len() < 3 {
                eprintln!("Error: validate requires a shader path or directory");
                exit(1);
            }
            if !validate_path(Path::new(&args[2])) {
                exit(1);
            }
        }
        "rhai" | "test-rhai" => {
            if args.len() < 3 {
                eprintln!("Error: {} requires a script path or directory", args[1]);
                exit(1);
            }
            if !test_rhai_path(Path::new(&args[2])) {
                exit(1);
            }
        }
        "bench" => {
            let dir = if args.len() >= 3 {
                Path::new(&args[2])
            } else {
                Path::new("shaders")
            };
            run_bench(dir);
        }
        "verify" => {
            let ok_shaders = validate_path(Path::new("shaders"));
            let ok_scripts = if Path::new("scripts").exists() {
                test_rhai_path(Path::new("scripts"))
            } else {
                true
            };
            if !(ok_shaders && ok_scripts) {
                exit(1);
            }
            println!("VERIFY: All checks passed.");
        }
        "info" => {
            println!("Shroud Engine v0.1.0");
            println!("Target Architecture: WebGPU Shading Language (WGSL) compute pipelines");
            println!("Scripting & Glue Stack: Rhai / Rust 2021 / C ABI / Python 3");
            println!("Status: Nominal");
        }
        _ => {
            print_usage();
            exit(1);
        }
    }
}
