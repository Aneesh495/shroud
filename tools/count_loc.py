#!/usr/bin/env python3
"""
Language Breakdown and Line-of-Code Counter
Scans repository and reports exact SLOC counts across HLSL, C++, Rhai, Python, and docs.

Copyright (c) 2026 Aneesh Krishna Parthasarathy
Licensed under the ISC License.
"""

import os
import sys

EXT_MAP = {
    ".hlsl": "HLSL",
    ".hlsli": "HLSL Header",
    ".cpp": "C++",
    ".cxx": "C++",
    ".cc": "C++",
    ".hpp": "C++ Header",
    ".h": "C/C++ Header",
    ".rhai": "Rhai",
    ".py": "Python",
    ".md": "Markdown",
    ".toml": "TOML",
    ".yml": "YAML",
    ".yaml": "YAML",
    ".txt": "CMake/Text",
}

EXCLUDE_DIRS = {".git", "target", "build", "dist", "__pycache__", "bin"}

def count_repository_loc(root_dir):
    stats = {}
    hlsl_total = 0
    total = 0
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            lang = EXT_MAP.get(ext, None)
            if not lang:
                continue
            fpath = os.path.join(root, f)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as fp:
                    count = len(fp.readlines())
                    stats[lang] = stats.get(lang, 0) + count
                    total += count
                    if "HLSL" in lang:
                        hlsl_total += count
            except Exception:
                pass
    return stats, total, hlsl_total

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    stats, total, hlsl_total = count_repository_loc(root)
    print("================================================================================")
    print("  SHROUD LINE-OF-CODE REPOSITORY BREAKDOWN")
    print("================================================================================")
    for lang, lines in sorted(stats.items(), key=lambda x: -x[1]):
        pct = (lines / total) * 100.0 if total > 0 else 0.0
        print(f"  {lang:<14} : {lines:>7} lines ({pct:5.1f}%)")
    print("--------------------------------------------------------------------------------")
    print(f"  RAW HLSL TOTAL : {hlsl_total:>7} lines")
    print(f"  PROJECT TOTAL  : {total:>7} lines (100.0%)")
    print("================================================================================")

if __name__ == "__main__":
    main()
