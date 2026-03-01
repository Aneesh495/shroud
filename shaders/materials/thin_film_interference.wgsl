// Shroud :: Multi-Layer Wave Optical Thin-Film Interference
// Evaluates phase differences and chromatic interference for anti-reflective coatings.

const PI: f32 = 3.14159265358979323846;

struct ThinFilmUniforms {
    film_thickness_nm: f32,
    film_ior: f32,
    substrate_ior: f32,
    ambient_ior: f32,
};

fn thin_film_spectral_reflectance(
    wavelength_nm: f32,
    cos_theta_0: f32,
    params: ThinFilmUniforms
) -> f32 {
    let n0 = params.ambient_ior;
    let n1 = params.film_ior;
    let n2 = params.substrate_ior;

    let sin_theta_1 = (n0 / n1) * sqrt(max(0.0, 1.0 - cos_theta_0 * cos_theta_0));
    if (sin_theta_1 >= 1.0) {
        return 1.0; // TIR
    }
    let cos_theta_1 = sqrt(max(0.0, 1.0 - sin_theta_1 * sin_theta_1));

    let sin_theta_2 = (n1 / n2) * sin_theta_1;
    let cos_theta_2 = sqrt(max(0.0, 1.0 - sin_theta_2 * sin_theta_2));

    // Fresnel reflection at interface 0-1
    let r01 = (n0 * cos_theta_0 - n1 * cos_theta_1) / (n0 * cos_theta_0 + n1 * cos_theta_1);
    // Fresnel reflection at interface 1-2
    let r12 = (n1 * cos_theta_1 - n2 * cos_theta_2) / (n1 * cos_theta_1 + n2 * cos_theta_2);

    // Optical path difference phase shift delta
    let delta = (4.0 * PI * n1 * params.film_thickness_nm * cos_theta_1) / wavelength_nm;

    let num = r01 * r01 + r12 * r12 + 2.0 * r01 * r12 * cos(delta);
    let den = 1.0 + r01 * r01 * r12 * r12 + 2.0 * r01 * r12 * cos(delta);

    return clamp(num / max(1e-7, den), 0.0, 1.0);
}

@group(0) @binding(0) var<uniform> u_thin_film: ThinFilmUniforms;
@group(0) @binding(1) var<storage, read> s_incident_cos: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_rgb_reflectance: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_thin_film_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_incident_cos)) {
        return;
    }

    let cos_t = s_incident_cos[idx];
    let r_red = thin_film_spectral_reflectance(650.0, cos_t, u_thin_film);
    let r_green = thin_film_spectral_reflectance(532.0, cos_t, u_thin_film);
    let r_blue = thin_film_spectral_reflectance(450.0, cos_t, u_thin_film);

    s_rgb_reflectance[idx] = vec4<f32>(r_red, r_green, r_blue, 1.0);
}
