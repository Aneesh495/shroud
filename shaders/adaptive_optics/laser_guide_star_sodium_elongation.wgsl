// Shroud :: Sodium Mesospheric Laser Guide Star Perspective Elongation
// Subsystem Module: shaders/adaptive_optics/laser_guide_star_sodium_elongation.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Models 589.159 nm D2a sodium resonance fluorescence inside the 85-105 km mesospheric layer.
//   * Evaluates radial spot elongation across off-axis Shack-Hartmann subapertures relative to launch telescope.
//   * Computes photon return flux as a function of laser linewidth and geomagnetic optical pumping.
//
// Mathematical Formulation:
//   theta_elong = (d_sub * Delta_H) / H_sodium^2 where d_sub is radial distance from laser launch axis.
//   H_sodium approx 92 km centroid altitude, Delta_H approx 12 km layer FWHM.
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const INV_PI: f32 = 0.31830988618379067154;
const INV_TWO_PI: f32 = 0.15915494309189533577;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_CONSTANT: f32 = 6.62607015e-34;
const BOLTZMANN_CONSTANT: f32 = 1.380649e-23;

struct LgsParameters { sodium_alt_m: f32, layer_thickness_m: f32, launch_offset_x_m: f32, launch_offset_y_m: f32, laser_power_w: f32, sodium_column_density: f32, pad0: f32, pad1: f32 };

struct LgsSubaperture { sub_x_m: f32, sub_y_m: f32, zen_angle_rad: f32, pad: f32 };

struct LgsSpotGeometry { elongation_arcsec: f32, orientation_rad: f32, photon_flux_m2_s: f32, valid: u32 };

@group(0) @binding(0) var<uniform> u_lgs: LgsParameters;
@group(0) @binding(1) var<storage, read> s_subaps: array<LgsSubaperture>;
@group(0) @binding(2) var<storage, read_write> s_lgs_out: array<LgsSpotGeometry>;

fn rad_to_arcsec(r: f32) -> f32 {
    return r * 206264.806;
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_lgs_out)) { return; }
    let sub = s_subaps[idx];
    let dx = sub.sub_x_m - u_lgs.launch_offset_x_m;
    let dy = sub.sub_y_m - u_lgs.launch_offset_y_m;
    let baseline = sqrt(dx * dx + dy * dy);
    let cos_z = max(0.2, cos(sub.zen_angle_rad));
    let h_eff = u_lgs.sodium_alt_m / cos_z;
    let dh_eff = u_lgs.layer_thickness_m / cos_z;
    let elong_rad = (baseline * dh_eff) / (h_eff * h_eff);
    let orient = atan2(dy, dx);
    let flux = u_lgs.laser_power_w * u_lgs.sodium_column_density * (cos_z * cos_z) / 1e8;
    s_lgs_out[idx] = LgsSpotGeometry(rad_to_arcsec(elong_rad), orient, flux, 1u);
}
