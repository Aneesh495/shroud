// Shroud :: Koehler Theory Aerosol Droplet Swelling
// Subsystem: atmospheric_microphysics :: hygroscopic_swelling_koehler
// Physically derived WebGPU compute kernel with numerical stability bounds.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_H: f32 = 6.62607015e-34;
const BOLTZMANN_K: f32 = 1.380649e-23;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;

/*
 * Theoretical and Mathematical Formulation for Koehler Theory Aerosol Droplet Swelling:
 * Calculates aerosol equilibrium droplet radius under variable relative humidity RH:
 * r(RH) = r_dry * (1 + kappa * (a_w / (1 - a_w)))^(1/3)
 * where kappa is the Petters and Kreidenweis hygroscopicity parameter.
 */

struct KoehlerParams { kappa: f32, r_dry_um: f32, temp_k: f32, padding: f32, };

@group(0) @binding(0) var<storage, read> s_input_data: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> s_output_data: array<vec4<f32>>; // Output payload

fn calculate_swollen_radius(r_dry: f32, rh: f32, kappa: f32) -> f32 {

    let a_w = clamp(rh, 0.01, 0.999);

    let growth_factor = pow(1.0 + kappa * (a_w / (1.0 - a_w)), 0.3333333);

    return r_dry * growth_factor;

}

@compute @workgroup_size(64, 1, 1)
fn cs_main_hygroscopic_swelling_koehler(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_output_data) || idx >= arrayLength(&s_input_data)) {
        return;
    }
    let in_val = s_input_data[idx];
    let r_swollen = calculate_swollen_radius(in_val.x, in_val.y, 0.3);
    let result = vec4<f32>(r_swollen, in_val.x, in_val.y, 1.0);
    s_output_data[idx] = result;
}
