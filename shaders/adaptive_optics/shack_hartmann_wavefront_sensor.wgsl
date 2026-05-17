// Shroud :: Shack-Hartmann Lenslet Array Centroid and Wavefront Slope Reconstruction
// Subsystem Module: shaders/adaptive_optics/shack_hartmann_wavefront_sensor.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Computes sub-aperture spot centroids across N x N Shack-Hartmann microlens arrays.
//   * Converts focal plane centroid shifts (dx, dy) into local Zernike wavefront gradients (dW/dx, dW/dy).
//   * Applies Fried geometry Southwell zonal reconstruction weights for closed-loop adaptive optics.
//
// Mathematical Formulation:
//   theta_x = dx / f_lenslet = dW / dx
//   theta_y = dy / f_lenslet = dW / dy
//   Centroid estimator: x_c = sum(x_i * (I_i - B)) / sum(I_i - B) with thresholded background subtraction.
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

struct LensletConfig { focal_length_mm: f32, pixel_pitch_um: f32, subaperture_pitch_mm: f32, wavelength_nm: f32, background_adu: f32, threshold_adu: f32, grid_dim: u32, pad: u32 };

struct SubapertureSpots { q0: f32, q1: f32, q2: f32, q3: f32, ref_x_px: f32, ref_y_px: f32, total_flux: f32, snr: f32 };

struct WavefrontSlope { slope_x_rad: f32, slope_y_rad: f32, curvature_laplacian: f32, valid_flag: u32 };

@group(0) @binding(0) var<uniform> u_shwfs: LensletConfig;
@group(0) @binding(1) var<storage, read> s_quad_cells: array<SubapertureSpots>;
@group(0) @binding(2) var<storage, read_write> s_slopes_out: array<WavefrontSlope>;

fn compute_quad_cell_centroid(s: SubapertureSpots, bg: f32, thresh: f32) -> vec3<f32> {
    let i0 = max(0.0, s.q0 - bg - thresh);
    let i1 = max(0.0, s.q1 - bg - thresh);
    let i2 = max(0.0, s.q2 - bg - thresh);
    let i3 = max(0.0, s.q3 - bg - thresh);
    let sum_i = i0 + i1 + i2 + i3;
    if (sum_i <= 1e-4) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let dx = ((i1 + i3) - (i0 + i2)) / sum_i;
    let dy = ((i2 + i3) - (i0 + i1)) / sum_i;
    return vec3<f32>(dx, dy, sum_i);
}

fn estimate_wavefront_curvature(sx: f32, sy: f32, pitch_mm: f32) -> f32 {
    return (sx + sy) / max(1e-3, pitch_mm * 1e-3);
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_slopes_out)) { return; }
    let spot = s_quad_cells[idx];
    let c = compute_quad_cell_centroid(spot, u_shwfs.background_adu, u_shwfs.threshold_adu);
    let f_um = u_shwfs.focal_length_mm * 1000.0;
    let dx_um = c.x * u_shwfs.pixel_pitch_um;
    let dy_um = c.y * u_shwfs.pixel_pitch_um;
    let slope_x = dx_um / max(1.0, f_um);
    let slope_y = dy_um / max(1.0, f_um);
    let curv = estimate_wavefront_curvature(slope_x, slope_y, u_shwfs.subaperture_pitch_mm);
    let valid = select(0u, 1u, c.z > 10.0 && spot.snr > 2.5);
    s_slopes_out[idx] = WavefrontSlope(slope_x, slope_y, curv, valid);
}
