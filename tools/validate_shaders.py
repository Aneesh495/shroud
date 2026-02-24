#!/usr/bin/env python3
"""
Automated HLSL Shader Compilation and SPIR-V Validation Tool
Validates all HLSL compute, vertex, and pixel shaders using glslangValidator.

Copyright (c) 2026 Aneesh Krishna Parthasarathy
Licensed under the ISC License.
"""

import os
import sys
import subprocess
import tempfile

def validate_shader(glslang_path, file_path, entry_point="CSMain", stage="comp"):
    with tempfile.NamedTemporaryFile(suffix=".spv", delete=True) as tmp_out:
        cmd = [
            glslang_path,
            "-D",
            "-e", entry_point,
            "-V",
            "-S", stage,
            "--hlsl-enable-16bit-types",
            "-o", tmp_out.name,
            file_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        return res.returncode == 0, res.stdout, res.stderr

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    shaders_dir = os.path.join(root, "shaders")
    
    glslang = subprocess.run(["which", "glslangValidator"], capture_output=True, text=True).stdout.strip()
    if not glslang:
        print("ERROR: glslangValidator not found on PATH.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Using shader compiler: {glslang}")
    
    # We will discover all .hlsl files
    hlsl_files = []
    for dirpath, _, filenames in os.walk(shaders_dir):
        for f in filenames:
            if f.endswith(".hlsl"):
                hlsl_files.append(os.path.join(dirpath, f))
                
    if not hlsl_files:
        print("No .hlsl files found yet in shaders directory.")
        return
        
    print(f"Validating {len(hlsl_files)} HLSL shaders...")
    failed = 0
    passed = 0
    
    for fpath in sorted(hlsl_files):
        rel = os.path.relpath(fpath, root)
        # Determine entry point and stage from file content or naming
        entry = "CSMain"
        stage = "comp"
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
                if "VSMain" in content and "PSMain" in content:
                    # Test both VS and PS
                    ok_vs, out_vs, err_vs = validate_shader(glslang, fpath, "VSMain", "vert")
                    ok_ps, out_ps, err_ps = validate_shader(glslang, fpath, "PSMain", "frag")
                    if ok_vs and ok_ps:
                        passed += 1
                        print(f"  [PASS] {rel} (VSMain + PSMain)")
                    else:
                        failed += 1
                        print(f"  [FAIL] {rel} (VSMain: {ok_vs}, PSMain: {ok_ps})", file=sys.stderr)
                        if not ok_vs: print(f"    VS error: {err_vs.strip()}", file=sys.stderr)
                        if not ok_ps: print(f"    PS error: {err_ps.strip()}", file=sys.stderr)
                    continue
                elif "VSMain" in content:
                    entry = "VSMain"
                    stage = "vert"
                elif "PSMain" in content:
                    entry = "PSMain"
                    stage = "frag"
                elif "CSMain" in content:
                    entry = "CSMain"
                    stage = "comp"
        except Exception as e:
            print(f"  [ERROR] Could not read {rel}: {e}", file=sys.stderr)
            failed += 1
            continue
            
        ok, out, err = validate_shader(glslang, fpath, entry, stage)
        if ok:
            passed += 1
            print(f"  [PASS] {rel} ({entry}:{stage})")
        else:
            failed += 1
            print(f"  [FAIL] {rel} ({entry}:{stage})", file=sys.stderr)
            print(f"    {err.strip()}", file=sys.stderr)
            
    print("--------------------------------------------------------------------------------")
    print(f"Summary: {passed} passed, {failed} failed out of {len(hlsl_files)} shaders.")
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
