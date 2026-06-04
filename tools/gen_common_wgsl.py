#!/usr/bin/env python3
"""
Generates shaders/common/ WGSL foundation modules (constants, types, math, rng, colorimetry).
All kernels are 100% standalone Naga-validatable WGSL compute units.

Copyright (c) 2026 Aneesh Krishna Parthasarathy
Licensed under the ISC License.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
COMMON_DIR = ROOT_DIR / "shaders" / "common"
COMMON_DIR.mkdir(parents=True, exist_ok=True)


def write_wgsl(name: str, src: str):
    p = COMMON_DIR / name
    p.write_text(src.strip() + "\n", encoding="utf-8")
    print(f"Wrote {p.relative_to(ROOT_DIR)} ({len(src.splitlines())} lines)")


CONSTANTS_WGSL = """// Shroud :: Common Physical & Optical Constants
// Module: shaders/common/constants.wgsl
// CODATA 2022 SI fundamental constants and standard CIE colorimetric anchors.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const INV_PI: f32 = 0.31830988618379067154;
const INV_FOUR_PI: f32 = 0.07957747154594766788;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_CONSTANT: f32 = 6.62607015e-34;
const BOLTZMANN_CONSTANT: f32 = 1.380649e-23;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;

@group(0) @binding(0) var<storage, read_write> s_constants_verify: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&s_constants_verify)) {
        s_constants_verify[idx] = vec4<f32>(PI, TWO_PI, INV_PI, SPEED_OF_LIGHT);
    }
}
"""

TYPES_WGSL = """// Shroud :: Common GPU Memory Layouts (std140/std430 Aligned)
// Module: shaders/common/types.wgsl
// Canonical GPU structs matching Rust `bytemuck::Pod` definitions in shroud-core.

struct RayGpu {
    origin: vec4<f32>,
    direction: vec4<f32>,
};

struct AabbGpu {
    min_pt: vec4<f32>,
    max_pt: vec4<f32>,
};

struct BvhNodeGpu {
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    left_or_first: u32,
    right_or_second: u32,
    prim_count: u32,
    pad: u32,
};

struct MediaPropertiesGpu {
    sigma_a: f32,
    sigma_s: f32,
    asymmetry_g: f32,
    pad: f32,
};

@group(0) @binding(0) var<storage, read> s_rays: array<RayGpu>;
@group(0) @binding(1) var<storage, read_write> s_bounds: array<AabbGpu>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&s_rays) && idx < arrayLength(&s_bounds)) {
        let r = s_rays[idx];
        let p_far = r.origin.xyz + normalize(r.direction.xyz) * 100.0;
        let lo = min(r.origin.xyz, p_far);
        let hi = max(r.origin.xyz, p_far);
        s_bounds[idx] = AabbGpu(vec4<f32>(lo, 1.0), vec4<f32>(hi, 1.0));
    }
}
"""

MATH_WGSL = """// Shroud :: Geometric & Orthonormal Basis Mathematical Utilities
// Module: shaders/common/math.wgsl
// Duff et al. (2017) branchless orthonormal basis and safe trigonometric helpers.

struct FrameBasis {
    tangent: vec4<f32>,
    bitangent: vec4<f32>,
    normal: vec4<f32>,
};

fn build_onb_duff(n_in: vec3<f32>) -> FrameBasis {
    let n = normalize(n_in);
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let b1 = vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let b2 = vec3<f32>(b, s + n.y * n.y * a, -n.y);
    return FrameBasis(vec4<f32>(b1, 0.0), vec4<f32>(b2, 0.0), vec4<f32>(n, 0.0));
}

@group(0) @binding(0) var<storage, read> s_normals_in: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> s_frames_out: array<FrameBasis>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&s_normals_in) && idx < arrayLength(&s_frames_out)) {
        s_frames_out[idx] = build_onb_duff(s_normals_in[idx].xyz);
    }
}
"""

RNG_WGSL = """// Shroud :: Stateless & State-Based GPU Pseudorandom Generators (PCG32 / Hash)
// Module: shaders/common/rng.wgsl

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32(seed: ptr<function, u32>) -> f32 {
    *seed = pcg_hash(*seed);
    return f32(*seed) * 2.3283064365386963e-10;
}

@group(0) @binding(0) var<storage, read_write> s_rng_samples: array<vec2<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&s_rng_samples)) {
        var state = idx * 1664525u + 1013904223u;
        let u0 = rand_f32(&state);
        let u1 = rand_f32(&state);
        s_rng_samples[idx] = vec2<f32>(u0, u1);
    }
}
"""

COLORIMETRY_WGSL = """// Shroud :: Spectral Colorimetry, ACES Filmic Tone Mapping, and sRGB Transfer
// Module: shaders/common/colorimetry.wgsl

fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

@group(0) @binding(0) var<storage, read> s_hdr_radiance: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> s_srgb_out: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&s_hdr_radiance) && idx < arrayLength(&s_srgb_out)) {
        let hdr = s_hdr_radiance[idx].rgb;
        let tm = aces_filmic(hdr);
        let srgb = linear_to_srgb(tm);
        s_srgb_out[idx] = vec4<f32>(srgb, 1.0);
    }
}
"""


def main():
    write_wgsl("constants.wgsl", CONSTANTS_WGSL)
    write_wgsl("types.wgsl", TYPES_WGSL)
    write_wgsl("math.wgsl", MATH_WGSL)
    write_wgsl("rng.wgsl", RNG_WGSL)
    write_wgsl("colorimetry.wgsl", COLORIMETRY_WGSL)


if __name__ == "__main__":
    main()
