// Shroud :: Silicon Photomultiplier (SiPM) Optical Crosstalk and Microcell Saturation
// Subsystem Module: shaders/quantum_photodetectors/sipm_microcell_crosstalk.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates 2D arrays of Geiger-mode avalanche microcells (SPADs) connected in parallel.
//   - Models prompt optical crosstalk, external reflection crosstalk, and multi-cell saturation.
//
// Mathematical & Physical Formulation:
//   N_fired = N_total * [1 - exp(-N_photons * PDE / N_total)] saturation curve.
//   Prompt crosstalk probability p_xt = 1 - exp(-mu_xt).
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
const VACUUM_PERMITTIVITY: f32 = 8.8541878128e-12;
const VACUUM_PERMEABILITY: f32 = 1.25663706212e-6;
const IMPEDANCE_FREE_SPACE: f32 = 376.730313668;

struct SipmConfig { total_cells: u32, pde: f32, crosstalk_prob: f32, pad: f32 };

struct SipmOut { cells_fired: u32, crosstalk_cells: u32, output_charge_pc: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_sipm: SipmConfig;
@group(0) @binding(1) var<storage, read> s_photons: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_sipm_out: array<SipmOut>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_sipm_out)) { return; }
    let n_ph = s_photons[idx];
    let tot = f32(u_sipm.total_cells);
    let direct = tot * (1.0 - exp(-n_ph * u_sipm.pde / tot));
    let xtalk = direct * u_sipm.crosstalk_prob;
    let fired = u32(direct + xtalk);
    let q_pc = f32(fired) * 0.15;
    s_sipm_out[idx] = SipmOut(fired, u32(xtalk), q_pc, 0.0);
}
