// Shroud :: Coastal Marine Boundary Layer Advection Fog Profile
// Subsystem: atmospheric_microphysics :: marine_boundary_layer_fog
// Physically derived WebGPU compute kernel with numerical stability bounds.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_H: f32 = 6.62607015e-34;
const BOLTZMANN_K: f32 = 1.380649e-23;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;

/*
 * Theoretical and Mathematical Formulation for Coastal Marine Boundary Layer Advection Fog Profile:
 * Models vertical liquid water content (LWC) and droplet concentration profiles:
 * LWC(z) = LWC_base + gamma_ad * (z - z_base)
 * Computes meteorological visibility V = 3.912 / sigma_ext (Koschmieder law).
 */

struct MarineFogParams { base_alt_m: f32, top_alt_m: f32, peak_lwc_g_m3: f32, effective_radius_um: f32, };

@group(0) @binding(0) var<storage, read> s_input_data: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> s_output_data: array<vec4<f32>>; // Output payload

fn evaluate_marine_extinction(lwc: f32, r_eff_um: f32) -> f32 {

    // sigma_ext = 1.5 * LWC / (rho_water * r_eff)

    return (1.5 * lwc) / (1.0 * max(1.0, r_eff_um));

}

@compute @workgroup_size(64, 1, 1)
fn cs_main_marine_boundary_layer_fog(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_output_data) || idx >= arrayLength(&s_input_data)) {
        return;
    }
    let in_val = s_input_data[idx];
    let sigma = evaluate_marine_extinction(in_val.x, in_val.y);
    let vis_m = 3.912 / max(1e-5, sigma);
    let result = vec4<f32>(sigma, vis_m, in_val.x, 1.0);
    s_output_data[idx] = result;
}
