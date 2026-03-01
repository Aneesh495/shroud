// Shroud :: Subsurface Scattering Dipole Diffusion Approximation
// BSSRDF diffusion model for semi-infinite participating slabs (Jensen et al. 2001).

const PI: f32 = 3.14159265358979323846;

struct DipoleMaterialParams {
    sigma_a: vec3<f32>,
    g: f32,
    sigma_s: vec3<f32>,
    eta: f32,
};

fn evaluate_dipole_profile_channel(r: f32, sigma_a: f32, sigma_s_prime: f32, eta: f32) -> f32 {
    let sigma_t_prime = sigma_a + sigma_s_prime;
    let alpha_prime = sigma_s_prime / max(1e-7, sigma_t_prime);
    let sigma_tr = sqrt(3.0 * sigma_a * sigma_t_prime);

    // Boundary condition parameter A
    let f_dr = -1.440 / (eta * eta) + 0.710 / eta + 0.668 + 0.0636 * eta;
    let a = (1.0 + f_dr) / (1.0 - f_dr);

    let z_pos = 1.0 / sigma_t_prime;
    let z_neg = z_pos * (1.0 + (4.0 / 3.0) * a);

    let d_pos = sqrt(r * r + z_pos * z_pos);
    let d_neg = sqrt(r * r + z_neg * z_neg);

    let term_pos = z_pos * (sigma_tr * d_pos + 1.0) * exp(-sigma_tr * d_pos) / (d_pos * d_pos * d_pos);
    let term_neg = z_neg * (sigma_tr * d_neg + 1.0) * exp(-sigma_tr * d_neg) / (d_neg * d_neg * d_neg);

    let rd = (alpha_prime / (4.0 * PI)) * (term_pos + term_neg);
    return max(0.0, rd);
}

@group(0) @binding(0) var<uniform> u_dipole: DipoleMaterialParams;
@group(0) @binding(1) var<storage, read> s_radii: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_diffuse_reflectance: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_dipole_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_radii)) {
        return;
    }

    let r = s_radii[idx];
    let s_s_prime = u_dipole.sigma_s * (1.0 - u_dipole.g);

    let rd_r = evaluate_dipole_profile_channel(r, u_dipole.sigma_a.r, s_s_prime.r, u_dipole.eta);
    let rd_g = evaluate_dipole_profile_channel(r, u_dipole.sigma_a.g, s_s_prime.g, u_dipole.eta);
    let rd_b = evaluate_dipole_profile_channel(r, u_dipole.sigma_a.b, s_s_prime.b, u_dipole.eta);

    s_diffuse_reflectance[idx] = vec4<f32>(rd_r, rd_g, rd_b, 1.0);
}
