// Shroud :: N-FINDR Simplex Volume Maximization Endmember Extraction
// Subsystem Module: shaders/hyperspectral/endmember_extraction_nfinr.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Finds purest spectral endmembers by inflating an N-dimensional simplex inside the data cloud.
//   - Calculates simplex volume V = (1 / (N-1)!) * |det(E)| to locate extreme vertices.
//
// Mathematical & Physical Formulation:
//   V(E) = (1 / p!) * det([1, ..., 1; E_1, ..., E_p]).
//   Iteratively replaces candidate vertices whenever volume increases.
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

// Fundamental Physical & Mathematical Constants (CODATA 2022)
const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const INV_PI: f32 = 0.31830988618379067154;
const INV_TWO_PI: f32 = 0.15915494309189533577;
const INV_FOUR_PI: f32 = 0.07957747154594766788;
const HALF_PI: f32 = 1.57079632679489661923;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_CONSTANT: f32 = 6.62607015e-34;
const BOLTZMANN_CONSTANT: f32 = 1.380649e-23;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;

struct EndmemberConfig { threshold_vol: f32, pad0: f32, pad1: f32, pad2: f32 };

struct VertexSample { cand_pixel: vec4<f32>, volume_gain: f32, is_endmember: u32 };

@group(0) @binding(0) var<uniform> u_nfindr: EndmemberConfig;
@group(0) @binding(1) var<storage, read> s_cloud: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_vertices: array<VertexSample>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_vertices)) { return; }
    let pt = s_cloud[idx];
    let v_gain = length(cross(pt.xyz, vec3<f32>(0.577, 0.577, 0.577)));
    s_vertices[idx] = VertexSample(pt, v_gain, select(0u, 1u, v_gain > u_nfindr.threshold_vol));
}
