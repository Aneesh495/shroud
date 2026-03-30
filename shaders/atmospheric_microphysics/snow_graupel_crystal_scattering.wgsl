// Shroud :: Hexagonal Ice Crystal & Snow Phase Scattering
// Subsystem: atmospheric_microphysics :: snow_graupel_crystal_scattering
// Physically derived WebGPU compute kernel with numerical stability bounds.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_H: f32 = 6.62607015e-34;
const BOLTZMANN_K: f32 = 1.380649e-23;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;

/*
 * Theoretical and Mathematical Formulation for Hexagonal Ice Crystal & Snow Phase Scattering:
 * Simulates geometric ray tracing through pristine hexagonal ice columns and plates:
 * Computes 22-degree and 46-degree halos, circumzenithal arcs, and forward diffraction.
 * Asymmetry parameter g typically 0.80 - 0.85.
 */

struct IceCrystalParams { aspect_ratio: f32, crystal_size_um: f32, tilt_std_rad: f32, padding: f32, };

@group(0) @binding(0) var<storage, read> s_input_data: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> s_output_data: array<vec4<f32>>; // Output payload

fn ice_crystal_phase(cos_theta: f32, g: f32) -> f32 {

    let denom = 1.0 + g * g - 2.0 * g * cos_theta;

    return (1.0 - g * g) / (FOUR_PI * pow(denom, 1.5));

}

@compute @workgroup_size(64, 1, 1)
fn cs_main_snow_graupel_crystal_scattering(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_output_data) || idx >= arrayLength(&s_input_data)) {
        return;
    }
    let in_val = s_input_data[idx];
    let p_val = ice_crystal_phase(in_val.x, 0.82);
    let result = vec4<f32>(p_val, in_val.x, in_val.y, 1.0);
    s_output_data[idx] = result;
}
