//! Command-line interface for Shroud: validation, benchmark, and Rhai testing.

use std::env;
use std::path::Path;
use std::process::exit;

fn print_usage() {
    println!("Shroud: High-Fidelity Multi-Spectral Optical Sensor Simulation and Wave Optics Engine");
    println!("Usage:");
    println!("  shroud validate <path-to-shader> [entry_point] [stage]");
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
                eprintln!("Error: validate requires a shader path");
                exit(1);
            }
            let path = Path::new(&args[2]);
            let entry = if args.len() >= 4 { &args[3] } else { "CSMain" };
            let stage = if args.len() >= 5 { &args[4] } else { "comp" };

            println!("Validating HLSL shader: {} ({}:{})...", path.display(), entry, stage);
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
            println!("Target Architecture: HLSL compute and graphics pipelines");
            println!("Glue Stack: C++20 / Rust 2021 / Rhai / Python 3");
            println!("Status: Nominal");
        }
        _ => {
            print_usage();
            exit(1);
        }
    }
}
