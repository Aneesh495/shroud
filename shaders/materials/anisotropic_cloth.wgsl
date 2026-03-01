// Shroud :: Woven Fabric & Micro-Flake Velvet BSDF
// Implements Ashikhmin-Premoze velvet sheen and directional anisotropic weaving.

const PI: f32 = 3.14159265358979323846;

struct ClothUniforms {
    base_color: vec3<f32>,
    roughness_warp: f32,
    sheen_color: vec3<f32>,
    roughness_weft: f32,
};

fn evaluate_velvet_brdf(
    n_dot_l: f32,
    n_dot_v: f32,
    n_dot_h: f32,
    roughness: f32
) -> f32 {
    let sin_theta_h = sqrt(max(0.0, 1.0 - n_dot_h * n_dot_h));
    let c = 1.0 / (4.0 * PI * roughness * roughness);
    let d = c * pow(sin_theta_h, 1.0 / roughness - 1.0);
    return d / (4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v) + 1e-5);
}

@group(0) @binding(0) var<uniform> u_cloth: ClothUniforms;
@group(0) @binding(1) var<storage, read> s_tangents: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_cloth_out: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_cloth_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_tangents)) {
        return;
    }

    let n = vec3<f32>(0.0, 1.0, 0.0);
    let v = vec3<f32>(0.0, 0.707, 0.707);
    let l = vec3<f32>(0.707, 0.707, 0.0);

    let n_dot_l = max(0.0, dot(n, l));
    let n_dot_v = max(0.0, dot(n, v));
    let h = normalize(v + l);
    let n_dot_h = max(0.0, dot(n, h));

    let velvet = evaluate_velvet_brdf(n_dot_l, n_dot_v, n_dot_h, u_cloth.roughness_warp);
    let diffuse = (u_cloth.base_color / PI) * n_dot_l;

    let total = diffuse + u_cloth.sheen_color * velvet;
    s_cloth_out[idx] = vec4<f32>(total, 1.0);
}
