// Shroud :: Microprismatic Retroreflection Model
// Models triple-reflection cube-corner prisms for automotive signs and road markers.

struct RetroreflectorParams {
    cube_size_mm: f32,
    efficiency: f32,
    acceptance_angle_deg: f32,
    asymmetry_dispersion: f32,
};

fn retroreflective_brdf(
    v: vec3<f32>,
    l: vec3<f32>,
    n: vec3<f32>,
    params: RetroreflectorParams
) -> vec3<f32> {
    let cos_incident = dot(n, v);
    if (cos_incident <= 0.0) {
        return vec3<f32>(0.0);
    }

    // Divergence angle between incoming light vector and outgoing view vector
    let cos_alpha = dot(l, v);
    let alpha_rad = acos(clamp(cos_alpha, -1.0, 1.0));

    // Peak response occurs when l matches v (exact retroreflection)
    let beam_width = 0.015; // Narrow retroreflective lobe in radians (approx 0.85 degrees)
    let retro_peak = exp(-(alpha_rad * alpha_rad) / (2.0 * beam_width * beam_width));

    let max_angle_rad = radians(params.acceptance_angle_deg);
    let incident_cutoff = smoothstep(cos(max_angle_rad), 1.0, cos_incident);

    let brdf_val = params.efficiency * retro_peak * incident_cutoff / (cos_incident + 1e-4);
    return vec3<f32>(brdf_val);
}

@group(0) @binding(0) var<uniform> u_retro: RetroreflectorParams;
@group(0) @binding(1) var<storage, read> s_ray_pairs: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_retro_returns: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_retroreflective_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx * 2u + 1u >= arrayLength(&s_ray_pairs)) {
        return;
    }

    let n = vec3<f32>(0.0, 1.0, 0.0);
    let v = s_ray_pairs[idx * 2u].xyz;
    let l = s_ray_pairs[idx * 2u + 1u].xyz;

    let res = retroreflective_brdf(v, l, n, u_retro);
    s_retro_returns[idx] = vec4<f32>(res, 1.0);
}
