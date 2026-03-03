// Shroud :: Rayleigh Molecular Atmospheric Scattering Phase Function
// Symmetric dipole scattering for clear molecular atmospheres.

const PI: f32 = 3.14159265358979323846;

fn rayleigh_phase_eval(cos_theta: f32) -> f32 {
    return (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);
}

@group(0) @binding(0) var<storage, read> s_cos_thetas: array<f32>;
@group(0) @binding(1) var<storage, read_write> s_phase_outputs: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn cs_rayleigh_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_cos_thetas)) {
        return;
    }
    let c = clamp(s_cos_thetas[idx], -1.0, 1.0);
    s_phase_outputs[idx] = rayleigh_phase_eval(c);
}
