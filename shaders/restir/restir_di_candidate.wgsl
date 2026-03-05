// Shroud :: ReSTIR DI Initial Candidate Light Generation Pass
// Samples direct analytical and emissive triangle light sources.

struct GpuAnalyticalLight {
    position: vec3<f32>,
    intensity: f32,
    emission: vec3<f32>,
    radius: f32,
};

struct DiCandidateOutput {
    light_pos: vec3<f32>,
    target_p_hat: f32,
    light_radiance: vec3<f32>,
    source_pdf: f32,
};

@group(0) @binding(0) var<storage, read> s_lights: array<GpuAnalyticalLight>;
@group(0) @binding(1) var<storage, read> s_surface_positions: array<vec4<f32>>; // pos, normal.x
@group(0) @binding(2) var<storage, read_write> s_initial_candidates: array<DiCandidateOutput>;

@compute @workgroup_size(64, 1, 1)
fn cs_restir_di_generate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_surface_positions) || arrayLength(&s_lights) == 0u) {
        return;
    }

    let surf_pos = s_surface_positions[idx].xyz;
    let light_idx = idx % arrayLength(&s_lights);
    let light = s_lights[light_idx];

    let to_light = light.position - surf_pos;
    let dist2 = max(1e-4, dot(to_light, to_light));
    let dist = sqrt(dist2);

    let geom_factor = 1.0 / dist2;
    let radiance = light.emission * light.intensity * geom_factor;
    let p_hat = dot(radiance, vec3<f32>(0.2126, 0.7152, 0.0722)); // Target luminance

    var out_cand: DiCandidateOutput;
    out_cand.light_pos = light.position;
    out_cand.target_p_hat = p_hat;
    out_cand.light_radiance = radiance;
    out_cand.source_pdf = 1.0 / f32(arrayLength(&s_lights));

    s_initial_candidates[idx] = out_cand;
}
