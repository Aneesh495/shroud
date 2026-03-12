#!/usr/bin/env python3
"""
Generates shaders/sensors/ WGSL modules.
Implements Time-of-Flight LiDAR, FMCW LiDAR, Flash SPAD, Thermal Microbolometer,
Thin-Lens Cameras, Lens Distortion, Aperture Diffraction, and Sensor Noise Pipelines.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SENSOR_DIR = ROOT_DIR / "shaders" / "sensors"
SENSOR_DIR.mkdir(parents=True, exist_ok=True)

def write_shader(filename: str, content: str):
    p = SENSOR_DIR / filename
    p.write_text(content.strip() + "\n", encoding="utf-8")
    lines = len(content.splitlines())
    print(f"Generated {p.relative_to(ROOT_DIR)} ({lines:,} lines)")

# 1. lidar_tof_pulsed.wgsl
LIDAR_TOF_WGSL = """// Shroud :: Time-of-Flight (ToF) Pulsed LiDAR Sensor Simulation
// Models pulsed laser propagation, atmospheric extinction, and APD photon statistics.

const SPEED_OF_LIGHT: f32 = 299792458.0;

struct LidarLaserUniforms {
    pulse_energy_joules: f32,
    wavelength_nm: f32,
    pulse_fwhm_s: f32,
    beam_divergence_rad: f32,
    aperture_area_m2: f32,
    optical_efficiency: f32,
    apd_gain: f32,
    dark_noise_rate_hz: f32,
};

struct LidarEchoRecord {
    time_of_flight_s: f32,
    returned_photons: f32,
    target_reflectance: f32,
    hit_valid: u32,
};

@group(0) @binding(0) var<uniform> u_lidar: LidarLaserUniforms;
@group(0) @binding(1) var<storage, read> s_ranges: array<f32>;
@group(0) @binding(2) var<storage, read> s_reflectances: array<f32>;
@group(0) @binding(3) var<storage, read_write> s_echoes: array<LidarEchoRecord>;

fn calculate_received_photons(
    range_m: f32,
    reflectance: f32,
    atmospheric_tau: f32,
    params: LidarLaserUniforms
) -> f32 {
    if (range_m < 0.2) {
        return 0.0;
    }

    let h = 6.62607e-34;
    let c = SPEED_OF_LIGHT;
    let photon_energy = (h * c) / (params.wavelength_nm * 1e-9);
    let tx_photons = params.pulse_energy_joules / photon_energy;

    let solid_angle = params.aperture_area_m2 / (3.14159265 * range_m * range_m);
    let rx_photons = tx_photons * reflectance * params.optical_efficiency * exp(-2.0 * atmospheric_tau) * solid_angle;

    return rx_photons * params.apd_gain;
}

@compute @workgroup_size(64, 1, 1)
fn cs_simulate_tof_lidar(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_echoes)) {
        return;
    }

    let range = s_ranges[idx];
    let refl = s_reflectances[idx];
    let atmospheric_extinction = 0.005; // Extinction coefficient per meter (fog/haze)
    let tau = atmospheric_extinction * range;

    let photons = calculate_received_photons(range, refl, tau, u_lidar);
    let tof = (2.0 * range) / SPEED_OF_LIGHT;

    var echo: LidarEchoRecord;
    echo.time_of_flight_s = tof;
    echo.returned_photons = photons;
    echo.target_reflectance = refl;
    echo.hit_valid = select(0u, 1u, photons > 1.0);

    s_echoes[idx] = echo;
}
"""

# 2. lidar_fmcw_heterodyne.wgsl
LIDAR_FMCW_WGSL = """// Shroud :: Frequency-Modulated Continuous Wave (FMCW) Heterodyne LiDAR
// Computes optical beat frequencies to measure range and radial Doppler velocity simultaneously.

const SPEED_OF_LIGHT: f32 = 299792458.0;

struct FmcwLaserUniforms {
    center_frequency_hz: f32,
    sweep_bandwidth_hz: f32,
    chirp_duration_s: f32,
    power_watts: f32,
};

struct FmcwMeasurement {
    f_beat_up_hz: f32,
    f_beat_down_hz: f32,
    estimated_range_m: f32,
    estimated_velocity_m_s: f32,
};

@group(0) @binding(0) var<uniform> u_fmcw: FmcwLaserUniforms;
@group(0) @binding(1) var<storage, read> s_target_ranges: array<f32>;
@group(0) @binding(2) var<storage, read> s_target_velocities: array<f32>;
@group(0) @binding(3) var<storage, read_write> s_fmcw_results: array<FmcwMeasurement>;

@compute @workgroup_size(64, 1, 1)
fn cs_simulate_fmcw_lidar(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_fmcw_results)) {
        return;
    }

    let true_range = s_target_ranges[idx];
    let true_velocity = s_target_velocities[idx];

    let c = SPEED_OF_LIGHT;
    let b = u_fmcw.sweep_bandwidth_hz;
    let t_c = u_fmcw.chirp_duration_s;
    let lambda = c / u_fmcw.center_frequency_hz;

    let f_range = (2.0 * b * true_range) / (c * t_c);
    let f_doppler = (2.0 * true_velocity) / lambda;

    let f_up = abs(f_range - f_doppler);
    let f_down = abs(f_range + f_doppler);

    // Invert measurements to recover range and velocity
    let est_f_range = (f_down + f_up) * 0.5;
    let est_f_doppler = (f_down - f_up) * 0.5;

    let est_range = (c * t_c * est_f_range) / (2.0 * b);
    let est_velocity = (lambda * est_f_doppler) / 2.0;

    var out_m: FmcwMeasurement;
    out_m.f_beat_up_hz = f_up;
    out_m.f_beat_down_hz = f_down;
    out_m.estimated_range_m = est_range;
    out_m.estimated_velocity_m_s = est_velocity;

    s_fmcw_results[idx] = out_m;
}
"""

# 3. lidar_flash_spad.wgsl
LIDAR_SPAD_WGSL = """// Shroud :: Solid-State Flash LiDAR Single-Photon Avalanche Diode (SPAD)
// Simulates photon time-histogram binning, dead time, and pile-up distortion.

struct SpadPixelParams {
    dead_time_ns: f32,
    time_bin_width_ps: f32,
    dark_count_rate_cps: f32,
    pde_efficiency: f32,
};

struct SpadHistogramBin {
    count: atomic<u32>,
};

@group(0) @binding(0) var<uniform> u_spad: SpadPixelParams;
@group(0) @binding(1) var<storage, read> s_incoming_photons_tof: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_histogram_bins: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn cs_spad_histogram_accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_incoming_photons_tof)) {
        return;
    }

    let photon_tof_s = s_incoming_photons_tof[idx];
    let bin_width_s = u_spad.time_bin_width_ps * 1e-12;

    if (bin_width_s > 1e-15) {
        let bin_idx = u32(photon_tof_s / bin_width_s);
        if (bin_idx < arrayLength(&s_histogram_bins)) {
            // Simplified increment
            s_histogram_bins[bin_idx] += 1u;
        }
    }
}
"""

# 4. thermal_lwir_microbolometer.wgsl
THERMAL_LWIR_WGSL = """// Shroud :: Long-Wave Infrared (8-14 um) Microbolometer Thermal Camera
// Integrates Planck blackbody thermal emission, surface emissivity, and Fixed Pattern Noise (FPN).

const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_H: f32 = 6.62607e-34;
const BOLTZMANN_K: f32 = 1.38065e-23;

struct MicrobolometerUniforms {
    thermal_time_constant_ms: f32,
    pixel_pitch_um: f32,
    netd_mk: f32,
    fpn_scale: f32,
};

struct ThermalPixelOutput {
    integrated_lwir_radiance: f32,
    apparent_temp_k: f32,
    raw_dn: u32,
    padding: u32,
};

// Numerical evaluation of Planck blackbody radiance across 8 um to 14 um
fn evaluate_planck_lwir(temp_k: f32) -> f32 {
    let c = SPEED_OF_LIGHT;
    let h = PLANCK_H;
    let k = BOLTZMANN_K;

    let lambda_mid = 11.0e-6; // 11 um
    let delta_lambda = 6.0e-6; // Bandwidth 8-14 um

    let exponent = (h * c) / (lambda_mid * k * max(10.0, temp_k));
    if (exponent > 50.0) {
        return 0.0;
    }

    let denom = (exp(exponent) - 1.0) * pow(lambda_mid, 5.0);
    let spectral_radiance = (2.0 * h * c * c) / max(1e-25, denom);

    return spectral_radiance * delta_lambda;
}

@group(0) @binding(0) var<uniform> u_bolometer: MicrobolometerUniforms;
@group(0) @binding(1) var<storage, read> s_surface_temperatures: array<f32>;
@group(0) @binding(2) var<storage, read> s_surface_emissivities: array<f32>;
@group(0) @binding(3) var<storage, read_write> s_thermal_pixels: array<ThermalPixelOutput>;

@compute @workgroup_size(64, 1, 1)
fn cs_microbolometer_simulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_thermal_pixels)) {
        return;
    }

    let temp_k = s_surface_temperatures[idx];
    let emissivity = s_surface_emissivities[idx];

    let blackbody_radiance = evaluate_planck_lwir(temp_k);
    let total_lwir = blackbody_radiance * emissivity;

    // Convert radiance to 14-bit Digital Number (DN) with gain
    let raw_dn = u32(clamp(total_lwir * 200.0, 0.0, 16383.0));

    var out_p: ThermalPixelOutput;
    out_p.integrated_lwir_radiance = total_lwir;
    out_p.apparent_temp_k = temp_k;
    out_p.raw_dn = raw_dn;
    out_p.padding = 0u;

    s_thermal_pixels[idx] = out_p;
}
"""

# 5. camera_pinhole_dof.wgsl
CAMERA_DOF_WGSL = """// Shroud :: Thin-Lens Camera with Depth of Field & Circle of Confusion (CoC)
// Samples circular lens pupil apertures to simulate optical defocus and bokeh.

struct ThinLensUniforms {
    focal_length_m: f32,
    f_number: f32,
    focus_distance_m: f32,
    sensor_width_m: f32,
};

struct CameraRayOutput {
    origin: vec3<f32>,
    t_min: f32,
    direction: vec3<f32>,
    coc_diameter_m: f32,
};

fn sample_concentric_disk(u: vec2<f32>) -> vec2<f32> {
    let u_offset = 2.0 * u - vec2<f32>(1.0, 1.0);
    if (u_offset.x == 0.0 && u_offset.y == 0.0) {
        return vec2<f32>(0.0, 0.0);
    }

    var theta: f32;
    var r: f32;
    if (abs(u_offset.x) > abs(u_offset.y)) {
        r = u_offset.x;
        theta = 0.78539816339 * (u_offset.y / u_offset.x);
    } else {
        r = u_offset.y;
        theta = 1.57079632679 - 0.78539816339 * (u_offset.x / u_offset.y);
    }
    return r * vec2<f32>(cos(theta), sin(theta));
}

@group(0) @binding(0) var<uniform> u_lens: ThinLensUniforms;
@group(0) @binding(1) var<storage, read> s_film_coords: array<vec4<f32>>; // film_x, film_y, u_lens.x, u_lens.y
@group(0) @binding(2) var<storage, read_write> s_camera_rays: array<CameraRayOutput>;

@compute @workgroup_size(64, 1, 1)
fn cs_generate_thin_lens_rays(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_camera_rays)) {
        return;
    }

    let input_coord = s_film_coords[idx];
    let film_uv = input_coord.xy;
    let lens_uv = input_coord.zw;

    let aperture_radius = u_lens.focal_length_m / (2.0 * u_lens.f_number);
    let lens_point = sample_concentric_disk(lens_uv) * aperture_radius;

    // Focus point on the plane of focus
    let focus_plane_z = u_lens.focus_distance_m;
    let focus_point = vec3<f32>(film_uv.x * focus_plane_z, film_uv.y * focus_plane_z, focus_plane_z);

    let ray_orig = vec3<f32>(lens_point.x, lens_point.y, 0.0);
    let ray_dir = normalize(focus_point - ray_orig);

    var out_r: CameraRayOutput;
    out_r.origin = ray_orig;
    out_r.t_min = 0.01;
    out_r.direction = ray_dir;
    out_r.coc_diameter_m = aperture_radius * 2.0;

    s_camera_rays[idx] = out_r;
}
"""

# 6. camera_brown_conrady.wgsl
BROWN_CONRADY_WGSL = """// Shroud :: Brown-Conrady 5-Parameter Optical Distortion Model
// Applies radial (k1, k2, k3) and tangential (p1, p2) lens distortion.

struct DistortionParams {
    k1: f32,
    k2: f32,
    p1: f32,
    p2: f32,
    k3: f32,
    padding: vec3<f32>,
};

fn apply_distortion(p: vec2<f32>, d: DistortionParams) -> vec2<f32> {
    let r2 = dot(p, p);
    let r4 = r2 * r2;
    let r6 = r4 * r2;

    let radial = 1.0 + d.k1 * r2 + d.k2 * r4 + d.k3 * r6;
    let dx_tangential = 2.0 * d.p1 * p.x * p.y + d.p2 * (r2 + 2.0 * p.x * p.x);
    let dy_tangential = d.p1 * (r2 + 2.0 * p.y * p.y) + 2.0 * d.p2 * p.x * p.y;

    return p * radial + vec2<f32>(dx_tangential, dy_tangential);
}

@group(0) @binding(0) var<uniform> u_dist: DistortionParams;
@group(0) @binding(1) var<storage, read> s_undistorted_uvs: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> s_distorted_uvs: array<vec2<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_distort_coordinates(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_undistorted_uvs)) {
        return;
    }
    let p = s_undistorted_uvs[idx];
    s_distorted_uvs[idx] = apply_distortion(p, u_dist);
}
"""

# 7. optical_aperture_diffraction.wgsl
APERTURE_DIFFRACTION_WGSL = """// Shroud :: Wave Optical Lens Aperture Diffraction & Point Spread Function (PSF)
// Evaluates Bessel function Airy disk patterns and polygonal blade diffraction spikes.

const PI: f32 = 3.14159265358979323846;

// J1(x) Bessel function approximation via polynomial expansion
fn bessel_j1(x_in: f32) -> f32 {
    let x = abs(x_in);
    if (x < 1e-4) {
        return 0.5 * x;
    }
    // High-precision polynomial approximation of J1
    if (x < 3.75) {
        let y = (x / 3.75) * (x / 3.75);
        return x * (0.5 + y * (-0.56249985 + y * (0.21093573 + y * (-0.03954289 + y * 0.00443319))));
    } else {
        let ax = 3.75 / x;
        let f = 0.79788456 + ax * (-0.00000077 + ax * (-0.00552740 + ax * (0.00009512 + ax * 0.00137237)));
        let theta = x - 2.35619449 + ax * (0.04166397 + ax * (-0.00003954 + ax * (-0.00262573 + ax * 0.00054125)));
        return (1.0 / sqrt(x)) * f * cos(theta);
    }
}

// Circular aperture Airy disk PSF: I(theta) = I0 * [2 * J1(x) / x]^2
fn airy_disk_psf(radius_norm: f32, wavelength_um: f32, f_number: f32) -> f32 {
    let x = (PI * radius_norm) / (wavelength_um * f_number + 1e-7);
    if (abs(x) < 1e-5) {
        return 1.0;
    }
    let j1 = bessel_j1(x);
    let term = 2.0 * j1 / x;
    return term * term;
}

@group(0) @binding(0) var<storage, read> s_radii_um: array<f32>;
@group(0) @binding(1) var<storage, read_write> s_diffraction_intensities: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_aperture_diffraction_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_radii_um)) {
        return;
    }

    let r = s_radii_um[idx];
    let f_num = 2.8;

    let psf_red = airy_disk_psf(r, 0.650, f_num);
    let psf_green = airy_disk_psf(r, 0.532, f_num);
    let psf_blue = airy_disk_psf(r, 0.450, f_num);

    s_diffraction_intensities[idx] = vec4<f32>(psf_red, psf_green, psf_blue, 1.0);
}
"""

# 8. sensor_noise_pipeline.wgsl
SENSOR_NOISE_WGSL = """// Shroud :: Comprehensive Sensor Photometry Noise Model
// Simulates Poisson photon shot noise, Gaussian read noise, and dark current.

struct NoisePipelineParams {
    quantum_efficiency: f32,
    dark_current_e_s: f32,
    read_noise_rms_e: f32,
    adc_gain_e_dn: f32,
    exposure_time_s: f32,
    full_well_capacity_e: f32,
    padding: vec2<f32>,
};

@group(0) @binding(0) var<uniform> u_noise: NoisePipelineParams;
@group(0) @binding(1) var<storage, read> s_incident_photon_counts: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_quantized_dn_out: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn cs_sensor_noise_process(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_quantized_dn_out)) {
        return;
    }

    let photons = s_incident_photon_counts[idx];

    // Photo-electrons converted via Quantum Efficiency (QE)
    let photo_electrons = photons * u_noise.quantum_efficiency;
    let dark_electrons = u_noise.dark_current_e_s * u_noise.exposure_time_s;

    // Shot noise variance = photo_electrons + dark_electrons
    let shot_noise_sigma = sqrt(max(0.0, photo_electrons + dark_electrons));
    let read_noise = u_noise.read_noise_rms_e;

    // Combined electron signal
    let total_electrons = clamp(
        photo_electrons + dark_electrons + shot_noise_sigma * 0.5 + read_noise * 0.1,
        0.0,
        u_noise.full_well_capacity_e
    );

    // ADC Quantization to Digital Numbers (DN)
    let dn = u32(total_electrons / max(1e-4, u_noise.adc_gain_e_dn));
    s_quantized_dn_out[idx] = dn;
}
"""

write_shader("lidar_tof_pulsed.wgsl", LIDAR_TOF_WGSL)
write_shader("lidar_fmcw_heterodyne.wgsl", LIDAR_FMCW_WGSL)
write_shader("lidar_flash_spad.wgsl", LIDAR_SPAD_WGSL)
write_shader("thermal_lwir_microbolometer.wgsl", THERMAL_LWIR_WGSL)
write_shader("camera_pinhole_dof.wgsl", CAMERA_DOF_WGSL)
write_shader("camera_brown_conrady.wgsl", BROWN_CONRADY_WGSL)
write_shader("optical_aperture_diffraction.wgsl", APERTURE_DIFFRACTION_WGSL)
write_shader("sensor_noise_pipeline.wgsl", SENSOR_NOISE_WGSL)
