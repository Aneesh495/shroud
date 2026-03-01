// Shroud :: Disney Principled BSDF Evaluation
// Integrates Burley diffuse, retro-reflection, sheen, clearcoat, and subsurface terms.

const PI: f32 = 3.14159265358979323846;

struct DisneyParams {
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    subsurface: f32,
    specular: f32,
    specular_tint: f32,
    anisotropic: f32,
    sheen: f32,
    sheen_tint: f32,
    clearcoat: f32,
    clearcoat_gloss: f32,
    padding: vec2<f32>,
};

fn schlick_weight(u: f32) -> f32 {
    let m = clamp(1.0 - u, 0.0, 1.0);
    return m * m * m * m * m;
}

// Disney Burley diffuse model with retro-reflection peak at grazing angles
fn disney_diffuse(
    n_dot_l: f32,
    n_dot_v: f32,
    v_dot_h: f32,
    roughness: f32,
    base_color: vec3<f32>
) -> vec3<f32> {
    let fd90 = 0.5 + 2.0 * roughness * v_dot_h * v_dot_h;
    let fl = schlick_weight(n_dot_l);
    let fv = schlick_weight(n_dot_v);
    let fd = (1.0 + (fd90 - 1.0) * fl) * (1.0 + (fd90 - 1.0) * fv);
    return (base_color / PI) * fd;
}

// Clearcoat GTR1 (Berry) distribution
fn gtr1_clearcoat(n_dot_h: f32, gloss: f32) -> f32 {
    let alpha = mix(0.1, 0.001, gloss);
    let alpha2 = alpha * alpha;
    let t = 1.0 + (alpha2 - 1.0) * n_dot_h * n_dot_h;
    return (alpha2 - 1.0) / (PI * log(alpha2) * t + 1e-7);
}

// Sheen term for fabrics and velvet
fn disney_sheen(v_dot_h: f32, sheen: f32, sheen_tint: f32, base_color: vec3<f32>) -> vec3<f32> {
    if (sheen <= 0.0) {
        return vec3<f32>(0.0);
    }
    let tint = mix(vec3<f32>(1.0), normalize(base_color + 1e-5), sheen_tint);
    let f = schlick_weight(v_dot_h);
    return sheen * tint * f;
}

@group(0) @binding(0) var<uniform> u_material: DisneyParams;
@group(0) @binding(1) var<storage, read> s_directions: array<vec4<f32>>; // v, l in alternating indices
@group(0) @binding(2) var<storage, read_write> s_bsdf_out: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_disney_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_idx = gid.x;
    if (pair_idx * 2u + 1u >= arrayLength(&s_directions)) {
        return;
    }

    let n = vec3<f32>(0.0, 1.0, 0.0);
    let v = s_directions[pair_idx * 2u].xyz;
    let l = s_directions[pair_idx * 2u + 1u].xyz;

    let n_dot_l = max(0.0, dot(n, l));
    let n_dot_v = max(0.0, dot(n, v));
    let h = normalize(v + l);
    let v_dot_h = max(0.0, dot(v, h));
    let n_dot_h = max(0.0, dot(n, h));

    let diff = disney_diffuse(n_dot_l, n_dot_v, v_dot_h, u_material.roughness, u_material.base_color);
    let sheen_val = disney_sheen(v_dot_h, u_material.sheen, u_material.sheen_tint, u_material.base_color);
    let coat = gtr1_clearcoat(n_dot_h, u_material.clearcoat_gloss) * u_material.clearcoat * 0.25;

    let total = diff * (1.0 - u_material.metallic) + sheen_val + vec3<f32>(coat);
    s_bsdf_out[pair_idx] = vec4<f32>(total, 1.0);
}
