// Shroud :: London Equations Magnetic Field Exponential Screening
// Subsystem Module: shaders/cryogenic_superconducting/london_penetration_depth_screening.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Simulates Meissner-Ochsenfeld magnetic flux expulsion in superconductors.
//   - Solves grad^2 B = B / lambda_L^2 yielding B(x) = B_0 * exp(-x / lambda_L).
//
// Mathematical & Physical Formulation:
//   lambda_L(0) = sqrt(m / (mu_0 * n_s * e^2)) London penetration depth.
//   Gorter-Casimir two-fluid temperature dependence: lambda_L(T) = lambda_L(0) / sqrt(1 - (T/Tc)^4).
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

struct LondonConfig { lambda0_nm: f32, tc_k: f32, b0_gauss: f32, pad: f32 };

struct LondonOut { b_field_gauss: f32, lambda_t_nm: f32, super_current_density: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_lon: LondonConfig;
@group(0) @binding(1) var<storage, read> s_depths_nm: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> s_lon_out: array<LondonOut>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_lon_out)) { return; }
    let smp = s_depths_nm[idx];
    let x_nm = smp.x;
    let t_k = smp.y;
    let lam_t = u_lon.lambda0_nm / sqrt(max(1e-3, 1.0 - pow(t_k / u_lon.tc_k, 4.0)));
    let b = u_lon.b0_gauss * exp(-x_nm / lam_t);
    let j_s = (b / max(1e-3, lam_t * 1e-9));
    s_lon_out[idx] = LondonOut(b, lam_t, j_s, 0.0);
}
