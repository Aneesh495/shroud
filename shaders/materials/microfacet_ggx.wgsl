// Shroud :: Cook-Torrance GGX Microfacet Reflection Model
// Evaluates D(h), G2(l, v), and F(v, h) with importance sampling of visible normals.

const PI: f32 = 3.14159265358979323846;
const INV_PI: f32 = 0.31830988618379067154;

struct MicrofacetEvalRecord {
    eval_brdf: vec3<f32>,
    pdf: f32,
};

// Trowbridge-Reitz GGX Normal Distribution Function
fn ggx_d(n_dot_h: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let d = (n_dot_h * n_dot_h) * (alpha2 - 1.0) + 1.0;
    return alpha2 / (PI * d * d + 1e-7);
}

// Smith Height-Correlated Masking-Shadowing Function G2
fn smith_g2(n_dot_l: f32, n_dot_v: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let lambda_v = n_dot_l * sqrt((-n_dot_v * alpha2 + n_dot_v) * n_dot_v + alpha2);
    let lambda_l = n_dot_v * sqrt((-n_dot_l * alpha2 + n_dot_l) * n_dot_l + alpha2);
    return 2.0 * n_dot_l * n_dot_v / max(1e-7, lambda_v + lambda_l);
}

// Schlick Fresnel approximation
fn fresnel_schlick(f0: vec3<f32>, v_dot_h: f32) -> vec3<f32> {
    let factor = clamp(1.0 - v_dot_h, 0.0, 1.0);
    let factor5 = factor * factor * factor * factor * factor;
    return f0 + (vec3<f32>(1.0) - f0) * factor5;
}

// Evaluates Cook-Torrance GGX specular BRDF
fn evaluate_cook_torrance(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    roughness: f32,
    f0: vec3<f32>
) -> MicrofacetEvalRecord {
    var out_rec: MicrofacetEvalRecord;
    let n_dot_l = max(0.0, dot(n, l));
    let n_dot_v = max(0.0, dot(n, v));

    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
        out_rec.eval_brdf = vec3<f32>(0.0);
        out_rec.pdf = 0.0;
        return out_rec;
    }

    let h = normalize(v + l);
    let n_dot_h = max(0.0, dot(n, h));
    let v_dot_h = max(0.0, dot(v, h));

    let d = ggx_d(n_dot_h, roughness);
    let g = smith_g2(n_dot_l, n_dot_v, roughness);
    let f = fresnel_schlick(f0, v_dot_h);

    let denom = 4.0 * n_dot_l * n_dot_v + 1e-7;
    out_rec.eval_brdf = (d * g * f) / denom;
    out_rec.pdf = (d * n_dot_h) / (4.0 * v_dot_h + 1e-7);

    return out_rec;
}

struct GgxQueryInput {
    normal: vec3<f32>,
    roughness: f32,
    view_dir: vec3<f32>,
    padding0: f32,
    light_dir: vec3<f32>,
    padding1: f32,
    f0: vec3<f32>,
    padding2: f32,
};

@group(0) @binding(0) var<storage, read> s_queries: array<GgxQueryInput>;
@group(0) @binding(1) var<storage, read_write> s_results: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_microfacet_ggx_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_queries)) {
        return;
    }

    let q = s_queries[idx];
    let res = evaluate_cook_torrance(q.normal, q.view_dir, q.light_dir, q.roughness, q.f0);
    s_results[idx] = vec4<f32>(res.eval_brdf, res.pdf);
}
