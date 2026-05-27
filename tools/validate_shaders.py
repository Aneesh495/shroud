#!/usr/bin/env python3
"""
Automated Dual-Backend Shader Compilation and SPIR-V Validation Tool for Shroud.
1. Compiles all HLSL (.hlsl) shaders to SPIR-V via glslangValidator and verifies
   structural/type conformance using Khronos spirv-val.
2. Validates all WebGPU (.wgsl) runtime shaders via Naga (shroud-cli).

Copyright (c) 2026 Aneesh Krishna Parthasarathy
Licensed under the ISC License.
"""

import argparse
import concurrent.futures
import os
import subprocess
import sys
import tempfile


def detect_entry_and_stage(content):
    if "VSMain" in content and "PSMain" in content:
        return [("VSMain", "vert"), ("PSMain", "frag")]
    if "VSMain" in content:
        return [("VSMain", "vert")]
    if "PSMain" in content:
        return [("PSMain", "frag")]
    return [("CSMain", "comp")]


def validate_hlsl_one(glslang, spirv_val, root, fpath):
    rel = os.path.relpath(fpath, root)
    shaders_root = os.path.join(root, "shaders")
    include_dir = os.path.join(shaders_root, "include")
    file_dir = os.path.dirname(os.path.abspath(fpath))

    try:
        with open(fpath, "r", encoding="utf-8") as fp:
            content = fp.read()
    except Exception as exc:
        return False, rel, f"Could not read file: {exc}"

    targets = detect_entry_and_stage(content)
    for entry, stage in targets:
        with tempfile.NamedTemporaryFile(suffix=".spv", delete=True) as tmp_out:
            cmd = [
                glslang,
                "-D",
                f"-I{shaders_root}",
                f"-I{include_dir}",
                f"-I{file_dir}",
                "-e",
                entry,
                "-V",
                "-S",
                stage,
                "--hlsl-enable-16bit-types",
                "-o",
                tmp_out.name,
                fpath,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                msg = (res.stdout + "\n" + res.stderr).strip()
                return False, rel, f"glslangValidator ({entry}:{stage}): {msg}"

            if spirv_val:
                val_res = subprocess.run(
                    [spirv_val, tmp_out.name], capture_output=True, text=True
                )
                if val_res.returncode != 0:
                    msg = (val_res.stdout + "\n" + val_res.stderr).strip()
                    return False, rel, f"spirv-val ({entry}:{stage}): {msg}"

    return True, rel, ",".join(f"{e}:{s}" for e, s in targets)


def validate_hlsl_suite(root, quiet=False):
    shaders_dir = os.path.join(root, "shaders")
    glslang = subprocess.run(
        ["which", "glslangValidator"], capture_output=True, text=True
    ).stdout.strip()
    if not glslang:
        print("ERROR: glslangValidator not found on PATH.", file=sys.stderr)
        return False

    spirv_val = subprocess.run(
        ["which", "spirv-val"], capture_output=True, text=True
    ).stdout.strip()

    hlsl_files = []
    for dirpath, _, filenames in os.walk(shaders_dir):
        for fname in filenames:
            if fname.endswith(".hlsl"):
                hlsl_files.append(os.path.join(dirpath, fname))
    hlsl_files.sort()

    if not hlsl_files:
        print("No .hlsl files found in shaders/.")
        return True

    print(
        f"Validating {len(hlsl_files)} HLSL shaders via glslangValidator"
        + (" + spirv-val..." if spirv_val else "...")
    )
    passed = 0
    failed = 0
    failures = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(8, os.cpu_count() or 4)
    ) as pool:
        futs = [
            pool.submit(validate_hlsl_one, glslang, spirv_val, root, fp)
            for fp in hlsl_files
        ]
        for fut in concurrent.futures.as_completed(futs):
            ok, rel, info = fut.result()
            if ok:
                passed += 1
                if not quiet:
                    print(f"  [PASS] {rel} ({info})")
            else:
                failed += 1
                failures.append((rel, info))

    for rel, err in sorted(failures):
        print(f"  [FAIL] {rel}\n    {err}", file=sys.stderr)

    print("-" * 80)
    print(f"HLSL Summary: {passed} passed, {failed} failed out of {len(hlsl_files)} shaders.")
    return failed == 0


def validate_wgsl_suite(root):
    shaders_dir = os.path.join(root, "shaders")
    cmd = ["cargo", "run", "-q", "-p", "shroud-cli", "--", "validate", shaders_dir]
    res = subprocess.run(cmd, cwd=root)
    return res.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Validate Shroud HLSL and WGSL shaders")
    parser.add_argument("--hlsl-only", action="store_true", help="Only validate HLSL shaders")
    parser.add_argument("--wgsl-only", action="store_true", help="Only validate WGSL shaders")
    parser.add_argument("--quiet", action="store_true", help="Only print failures and summary")
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ok = True
    if not args.wgsl_only:
        ok = validate_hlsl_suite(root, quiet=args.quiet) and ok
    if not args.hlsl_only:
        ok = validate_wgsl_suite(root) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
