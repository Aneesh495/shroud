// Shroud :: Mie Marine Aerosol & Sea-Spray Phase Function
// Two-term Cornette-Shanks expansion with forward diffraction spike and backward glory.

const PI: f32 = 3.14159265358979323846;

struct MarineMieParams {
    g_forward: f32,
    g_backward: f32,
    forward_weight: f32,
    padding: f32,
};

fn cornette_shanks_term(cos_theta: f32, g: f32) -> f32 {
    let num = 3.0 * (1.0 - g * g) * (1.0 + cos_theta * cos_theta);
    let den = 8.0 * PI * (2.0 + g * g) * pow(1.0 + g * g - 2.0 * g * cos_theta, 1.5);
    return num / max(1e-7, den);
}

fn marine_aerosol_phase(cos_theta: f32, params: MarineMieParams) -> f32 {
    let pf = cornette_shanks_term(cos_theta, params.g_forward);
    let pb = cornette_shanks_term(cos_theta, params.g_backward);
    return params.forward_weight * pf + (1.0 - params.forward_weight) * pb;
}

@group(0) @binding(0) var<uniform> u_marine_params: MarineMieParams;
@group(0) @binding(1) var<storage, read> s_incident_cos: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_mie_results: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn cs_mie_marine_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_incident_cos)) {
        return;
    }
    let c = clamp(s_incident_cos[idx], -1.0, 1.0);
    s_mie_results[idx] = marine_aerosol_phase(c, u_marine_params);
}
