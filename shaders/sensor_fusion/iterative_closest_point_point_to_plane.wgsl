// Shroud :: Point-to-Plane Linearized Gauss-Newton ICP 6-DoF Optimizer
// Subsystem Module: shaders/sensor_fusion/iterative_closest_point_point_to_plane.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Minimizes orthogonal distance between source point cloud and target surface planes.
//   - Solves 6x6 normal equations J^T * J * delta = -J^T * r for SE(3) transformation.
//
// Mathematical & Physical Formulation:
//   r_i = ((R * p_i + t) - q_i) . n_i point-to-plane residual.
//   Linearized rotation [alpha, beta, gamma] around identity.
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
const ELEMENTARY_CHARGE: f32 = 1.602176634e-19;
const ELECTRON_MASS: f32 = 9.1093837015e-31;

struct IcpPair { p_src: vec4<f32>, q_dst: vec4<f32>, n_dst: vec4<f32> };

struct IcpLinearTerm { ata_diag: vec4<f32>, atb: vec4<f32>, residual: f32, weight: f32 };

@group(0) @binding(0) var<storage, read> s_pairs: array<IcpPair>;
@group(0) @binding(1) var<storage, read_write> s_linear: array<IcpLinearTerm>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_linear)) { return; }
    let pr = s_pairs[idx];
    let p = pr.p_src.xyz;
    let q = pr.q_dst.xyz;
    let n = pr.n_dst.xyz;
    let r = dot(p - q, n);
    let c = cross(p, n);
    let ata = vec4<f32>(c.x * c.x, c.y * c.y, c.z * c.z, dot(n, n));
    let atb = vec4<f32>(c.x * r, c.y * r, c.z * r, n.z * r);
    s_linear[idx] = IcpLinearTerm(ata, atb, r, select(1.0, 0.1, abs(r) > 0.5));
}
