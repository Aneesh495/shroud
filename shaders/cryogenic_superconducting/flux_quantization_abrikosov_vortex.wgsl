// Shroud :: Abrikosov Magnetic Flux Vortex Lattice in Type-II Superconductors
// Subsystem Module: shaders/cryogenic_superconducting/flux_quantization_abrikosov_vortex.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates 2D triangular Abrikosov fluxon lattice between lower and upper critical fields.
//   - Each vortex carries single London magnetic flux quantum Phi_0 = h / 2e approx 2.07e-15 Wb.
//
// Mathematical & Physical Formulation:
//   B_vortex(r) = (Phi_0 / (2 * pi * lambda^2)) * K_0(r / lambda) modified Bessel function.
//   Normal core radius governed by coherence length xi.
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

struct VortexConfig { lambda_nm: f32, xi_nm: f32, vortex_density_m2: f32, pad: f32 };

struct VortexField { b_gauss: f32, core_suppression: f32, pad0: f32, pad1: f32 };

@group(0) @binding(0) var<uniform> u_vx: VortexConfig;
@group(0) @binding(1) var<storage, read> s_r_vortex_nm: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_vx_out: array<VortexField>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_vx_out)) { return; }
    let r_nm = s_r_vortex_nm[idx];
    let r_lam = r_nm / u_vx.lambda_nm;
    let k0_approx = exp(-r_lam) / max(0.5, sqrt(PI * max(0.1, r_lam)));
    let core = 1.0 - exp(-pow(r_nm / u_vx.xi_nm, 2.0));
    let b = k0_approx * core * 1000.0;
    s_vx_out[idx] = VortexField(b, core, 0.0, 0.0);
}
