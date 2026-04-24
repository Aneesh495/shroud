// Shroud :: Multispectral Vegetation Indices (NDVI, EVI, NDRE, PRI) Evaluator
// Subsystem Module: shaders/hyperspectral/vegetation_indices_ndvi_pri.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Computes biophysical photosynthetic indices from Red, NIR, Red-Edge, and 531 nm bands.
//   - Estimates canopy greenness, nitrogen content, and xanthophyll cycle light use efficiency.
//
// Mathematical & Physical Formulation:
//   NDVI = (NIR - Red) / (NIR + Red), NDRE = (NIR - RE) / (NIR + RE).
//   PRI = (R531 - R570) / (R531 + R570) photochemical reflectance index.
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

struct VegBands { r_red: f32, r_nir: f32, r_re: f32, r_531: f32, r_570: f32, r_blue: f32, pad0: f32, pad1: f32 };

struct VegIndices { ndvi: f32, evi: f32, ndre: f32, pri: f32, chlor_index: f32, pad: f32 };

@group(0) @binding(0) var<storage, read> s_veg_in: array<VegBands>;
@group(0) @binding(1) var<storage, read_write> s_veg_out: array<VegIndices>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_veg_out)) { return; }
    let b = s_veg_in[idx];
    let ndvi = (b.r_nir - b.r_red) / max(1e-4, b.r_nir + b.r_red);
    let evi = 2.5 * (b.r_nir - b.r_red) / max(1e-4, b.r_nir + 6.0 * b.r_red - 7.5 * b.r_blue + 1.0);
    let ndre = (b.r_nir - b.r_re) / max(1e-4, b.r_nir + b.r_re);
    let pri = (b.r_531 - b.r_570) / max(1e-4, b.r_531 + b.r_570);
    let ci = (b.r_nir / max(1e-4, b.r_re)) - 1.0;
    s_veg_out[idx] = VegIndices(ndvi, evi, ndre, pri, ci, 0.0);
}
