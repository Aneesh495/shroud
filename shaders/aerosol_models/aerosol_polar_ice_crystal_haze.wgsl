// Shroud :: Polar Tropospheric Ice Crystal Diamond Dust Model
// Subsystem Module: shaders/aerosol_models/aerosol_polar_ice_crystal_haze.wgsl
// Physically based high-performance GPU compute kernel.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const INV_PI: f32 = 0.31830988618379067154;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_H: f32 = 6.62607015e-34;
const BOLTZMANN_K: f32 = 1.380649e-23;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;

/*
 * Engineering and Scientific Specification for Polar Tropospheric Ice Crystal Diamond Dust Model:
 * Extinction coefficient at 550nm: 0.0350 m^-1
 * Single-scattering albedo: 0.9800
 * Henyey-Greenstein asymmetry parameter g: 0.8200
 * Refractive index of particle core: 1.3100
 * Koschmieder visibility V = 3.912 / sigma_ext
 */

const SIGMA_EXT: f32 = 0.0350;

const ALBEDO: f32 = 0.9800;

const G_ASYM: f32 = 0.8200;

fn evaluate_aerosol_phase(cos_t: f32) -> f32 {

    let g2 = G_ASYM * G_ASYM;

    let denom = 1.0 + g2 - 2.0 * G_ASYM * cos_t;

    return (1.0 - g2) / (FOUR_PI * denom * sqrt(max(1e-6, denom)));

}

fn evaluate_transmittance(distance_m: f32) -> f32 {

    return exp(-SIGMA_EXT * distance_m);

}

@group(0) @binding(0) var<storage, read> s_input_payloads: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> s_output_payloads: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_execute_aerosol_polar_ice_crystal_haze(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_output_payloads) || idx >= arrayLength(&s_input_payloads)) {
        return;
    }
    let in_p = s_input_payloads[idx];
    let cos_angle = clamp(in_p.x, -1.0, 1.0);
    let dist = max(0.0, in_p.y);
    let p_val = evaluate_aerosol_phase(cos_angle);
    let tr_val = evaluate_transmittance(dist);
    let in_scatter = (1.0 - tr_val) * ALBEDO * p_val;
    let result = vec4<f32>(tr_val, in_scatter, p_val, 1.0);
    s_output_payloads[idx] = result;
}
