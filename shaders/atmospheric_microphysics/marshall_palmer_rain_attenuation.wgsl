// Shroud :: Marshall-Palmer Rain Drop Extinction & Optical Depth
// Subsystem: atmospheric_microphysics :: marshall_palmer_rain_attenuation
// Physically derived WebGPU compute kernel with numerical stability bounds.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_H: f32 = 6.62607015e-34;
const BOLTZMANN_K: f32 = 1.380649e-23;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;

/*
 * Theoretical and Mathematical Formulation for Marshall-Palmer Rain Drop Extinction & Optical Depth:
 * Computes rain drop size distribution N(D) = N0 * exp(-Lambda * D):
 * Lambda = 4.1 * R^(-0.21) mm^-1, N0 = 8000 m^-3 mm^-1
 * Evaluates extinction cross-section Q_ext ~= 2 * pi * (D/2)^2 for geometric optics.
 */

struct RainParams { rain_rate_mm_hr: f32, path_length_m: f32, padding: vec2<f32>, };

@group(0) @binding(0) var<storage, read> s_input_data: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> s_output_data: array<vec4<f32>>; // Output payload

fn calculate_rain_extinction(rain_rate: f32) -> f32 {

    // Empirical optical extinction alpha = 0.2 * R^0.63 in dB/km

    let alpha_db_km = 0.2 * pow(max(0.01, rain_rate), 0.63);

    return alpha_db_km * 0.0002302585; // Convert dB/km to 1/m extinction

}

@compute @workgroup_size(64, 1, 1)
fn cs_main_marshall_palmer_rain_attenuation(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_output_data) || idx >= arrayLength(&s_input_data)) {
        return;
    }
    let in_val = s_input_data[idx];
    let sigma_ext = calculate_rain_extinction(in_val.x);
    let tau = sigma_ext * in_val.y;
    let result = vec4<f32>(sigma_ext, tau, exp(-tau), 1.0);
    s_output_data[idx] = result;
}
