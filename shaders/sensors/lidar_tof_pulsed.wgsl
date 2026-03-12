// Shroud :: Time-of-Flight (ToF) Pulsed LiDAR Sensor Simulation
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
