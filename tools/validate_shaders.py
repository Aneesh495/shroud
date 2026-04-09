#!/usr/bin/env python3
"""
Automated WGSL Shader Validation Wrapper
Invokes the Rust Naga-backed shroud CLI to parse, typecheck, and validate every WGSL kernel.

Copyright (c) 2026 Aneesh Krishna Parthasarathy
Licensed under the ISC License.
"""

import os
import subprocess
import sys


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    shaders_dir = os.path.join(root, "shaders")

    cmd = ["cargo", "run", "-q", "-p", "shroud-cli", "--", "validate", shaders_dir]
    res = subprocess.run(cmd, cwd=root)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
