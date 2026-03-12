// Shroud :: Long-Wave Infrared (8-14 um) Microbolometer Thermal Camera
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
