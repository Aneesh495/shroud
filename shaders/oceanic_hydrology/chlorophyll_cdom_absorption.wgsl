// Shroud :: Pope-Fry Water Absorption and Morel Case 1 Bio-Optical Model
// Subsystem Module: shaders/oceanic_hydrology/chlorophyll_cdom_absorption.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates total marine absorption a(lambda) = a_w(lambda) + a_chl(lambda) + a_cdom(lambda).
//   - Combines Pope & Fry pure water spectrum with chlorophyll-a and humic acid CDOM decay.
//
// Mathematical & Physical Formulation:
//   a_cdom(lambda) = a_cdom(440) * exp(-S * (lambda - 440)) where S approx 0.014 nm^-1.
//   a_chl(lambda) = 0.06 * [Chl]^0.65 * a*_chl(lambda).
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

struct BioOpticsParams { chl_mg_m3: f32, a_cdom_440: f32, s_slope: f32, pad: f32 };

struct BioAbsorptionOut { a_pure_water: f32, a_chl: f32, a_cdom: f32, a_total: f32 };

@group(0) @binding(0) var<uniform> u_bio: BioOpticsParams;
@group(0) @binding(1) var<storage, read> s_lambdas: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_bio_out: array<BioAbsorptionOut>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_bio_out)) { return; }
    let l = s_lambdas[idx];
    let aw = 0.005 + 0.05 * pow(l / 600.0, 3.5);
    let achl = 0.06 * pow(u_bio.chl_mg_m3, 0.65) * exp(-0.5 * pow((l - 440.0) / 30.0, 2.0));
    let acdom = u_bio.a_cdom_440 * exp(-u_bio.s_slope * (l - 440.0));
    let atot = aw + achl + acdom;
    s_bio_out[idx] = BioAbsorptionOut(aw, achl, acdom, atot);
}
