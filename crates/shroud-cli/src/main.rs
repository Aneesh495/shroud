//! Command-line interface for Shroud: validation, benchmark, and Rhai testing.

use std::env;
use std::path::Path;
use std::process::exit;

fn print_usage() {
    println!("Shroud: High-Fidelity Multi-Spectral Optical Sensor Simulation and Wave Optics Engine");
    println!("Usage:");
    println!("  shroud validate <path-to-shader-or-dir> [entry_point] [stage]");
    println!("  shroud rhai <path-to-script>");
    println!("  shroud info");
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
            let path = Path::new(&args[2]);
            if path.is_dir() {
                println!("Validating WGSL shader directory: {}...", path.display());
                let results = shroud_validator::validate_wgsl_directory(path);
                let mut passed = 0;
                let mut failed = 0;
                for res in &results {
                    if res.is_valid {
                        passed += 1;
                    } else {
                        failed += 1;
                        eprintln!("FAIL: {}\n{}", res.file_path, res.errors.join("\n"));
                    }
                }
                println!("Validation summary: {} passed, {} failed out of {} WGSL shaders.",
                    passed, failed, results.len());
                if failed > 0 {
                    exit(1);
                } else {
                    println!("PASS: All WGSL shaders compiled and validated successfully via Naga.");
                }
            } else if path.extension().and_then(|s| s.to_str()) == Some("wgsl") {
                println!("Validating WGSL shader: {}...", path.display());
                match std::fs::read_to_string(path) {
                    Ok(source) => {
                        let res = shroud_validator::validate_wgsl_source(&path.display().to_string(), &source);
                        if res.is_valid {
                            println!("PASS: WGSL shader validated successfully. Entry points: {:?}", res.entry_points);
                        } else {
                            eprintln!("FAIL: WGSL validation failed:\n{}", res.errors.join("\n"));
                            exit(1);
                        }
                    }
                    Err(e) => {
                        eprintln!("Error reading file: {}", e);
                        exit(1);
                    }
                }
            } else {
                let entry = if args.len() >= 4 { &args[3] } else { "CSMain" };
                let stage = if args.len() >= 5 { &args[4] } else { "comp" };

                println!("Validating shader: {} ({}:{})...", path.display(), entry, stage);
                match shroud_validator::validate_hlsl_file(path, entry, stage) {
                    Ok(res) => {
                        if res.is_valid {
                            println!("PASS: Shader validated successfully.");
                        } else {
                            eprintln!("FAIL: Shader validation failed:\n{}", res.errors.join("\n"));
                            exit(1);
                        }
                    }
                    Err(e) => {
                        eprintln!("Error: {}", e);
                        exit(1);
                    }
                }
            }
        }
        "rhai" => {
            if args.len() < 3 {
                eprintln!("Error: rhai requires a script path");
                exit(1);
            }
            let script_path = Path::new(&args[2]);
            println!("Executing Rhai script: {}...", script_path.display());
            match std::fs::read_to_string(script_path) {
                Ok(script) => {
                    let runner = shroud_rhai::ShroudRhaiEngine::new();
                    match runner.eval_script::<rhai::Dynamic>(&script) {
                        Ok(_) => println!("PASS: Rhai script executed successfully."),
                        Err(e) => {
                            eprintln!("FAIL: Script execution error: {}", e);
                            exit(1);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error reading script: {}", e);
                    exit(1);
                }
            }
        }
        "info" => {
            println!("Shroud Engine v0.1.0");
            println!("Target Architecture: WebGPU Shading Language (WGSL) compute pipelines");
            println!("Glue Stack: C ABI / Rust 2021 / Rhai / Python 3");
            println!("Status: Nominal");
        }
        _ => {
            print_usage();
            exit(1);
        }
    }
}

