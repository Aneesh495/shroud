// Shroud :: Comprehensive Sensor Photometry Noise Model
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
