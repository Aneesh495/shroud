// Shroud :: Zernike Wavefront Aberrations and Optical Point Spread Function
// Subsystem Module: shaders/wave_optics/airy_disk_psf.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates circular Zernike polynomial expansions up to order 4.
//   - Computes diffraction-limited Airy disc intensity profile and Strehl ratio.
//
// Mathematical & Physical Formulation:
//   W(rho, phi) = sum_n sum_m c_n^m * R_n^m(rho) * cos(m * phi)
//   I(v) = [2 * J_1(v) / v]^2 where v = (2 * pi / lambda) * NA * r
//   Strehl S = exp(-sigma_phi^2) via Marechal approximation.
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

struct ZernikeCoeffs { c0: f32, c1: f32, c2: f32, c3: f32, c4: f32, c5: f32, c6: f32, c7: f32, c8: f32, na: f32, wavelength_nm: f32, pupil_radius_m: f32 };

struct PsfResult { intensity: f32, phase_rad: f32, encircled_energy: f32, strehl_ratio: f32 };

@group(0) @binding(0) var<uniform> u_zernike: ZernikeCoeffs;
@group(0) @binding(1) var<storage, read> s_pupil_coords: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_psf_out: array<PsfResult>;

fn eval_zernike(rho: f32, phi: f32, z: ZernikeCoeffs) -> f32 {
    let r2 = rho * rho;
    let r3 = r2 * rho;
    let r4 = r2 * r2;
    let p = z.c0;
    let tx = z.c1 * 2.0 * rho * cos(phi);
    let ty = z.c2 * 2.0 * rho * sin(phi);
    let def = z.c3 * sqrt(3.0) * (2.0 * r2 - 1.0);
    let a0 = z.c4 * sqrt(6.0) * r2 * cos(2.0 * phi);
    let a45 = z.c5 * sqrt(6.0) * r2 * sin(2.0 * phi);
    let cx = z.c6 * sqrt(8.0) * (3.0 * r3 - 2.0 * rho) * cos(phi);
    let cy = z.c7 * sqrt(8.0) * (3.0 * r3 - 2.0 * rho) * sin(phi);
    let sph = z.c8 * sqrt(5.0) * (6.0 * r4 - 6.0 * r2 + 1.0);
    return p + tx + ty + def + a0 + a45 + cx + cy + sph;
}

fn bessel_j1(x: f32) -> f32 {
    let ax = abs(x);
    if (ax < 1e-5) { return 0.5 * x; }
    let t = ax / 3.0;
    if (ax <= 3.0) {
        let t2 = t * t;
        let poly = x * (0.5 + t2 * (-0.56249985 + t2 * (0.21093573 + t2 * (-0.03954289 + t2 * 0.00443319))));
        return poly;
    } else {
        return sqrt(2.0 / (PI * ax)) * cos(ax - 0.75 * PI) * sign(x);
    }
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_pupil_coords)) { return; }
    let c = s_pupil_coords[idx];
    let rho = clamp(c.x, 0.0, 1.0);
    let phi = c.y;
    let r_dist = c.z;
    let w_um = eval_zernike(rho, phi, u_zernike);
    let lambda_um = u_zernike.wavelength_nm * 1e-3;
    let phase_rad = TWO_PI * (w_um / max(1e-4, lambda_um));
    let k = TWO_PI / (u_zernike.wavelength_nm * 1e-9);
    let v = k * u_zernike.na * (r_dist * 1e-6);
    let j1 = bessel_j1(v);
    let airy = select(1.0, (2.0 * j1 / max(1e-5, v)) * (2.0 * j1 / max(1e-5, v)), abs(v) > 1e-5);
    let strehl = exp(-min(4.0, phase_rad * phase_rad));
    let enc = clamp(1.0 - j1 * j1, 0.0, 1.0);
    s_psf_out[idx] = PsfResult(airy * strehl, phase_rad, enc, strehl);
}
