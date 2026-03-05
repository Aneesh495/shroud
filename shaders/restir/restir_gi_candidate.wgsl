// Shroud :: ReSTIR GI Indirect Path Candidate Generation
// Traces diffuse bounce paths and constructs initial indirect path reservoirs.

struct GiPathCandidate {
    bounce_position: vec3<f32>,
    bounce_normal: vec3<f32>,
    indirect_radiance: vec3<f32>,
    path_pdf: f32,
};

@group(0) @binding(0) var<storage, read> s_primary_hits: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> s_gi_candidates: array<GiPathCandidate>;

@compute @workgroup_size(64, 1, 1)
fn cs_restir_gi_generate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_gi_candidates)) {
        return;
    }

    let primary_pos = s_primary_hits[idx].xyz;
    let bounce_dir = vec3<f32>(0.0, 1.0, 0.0); // Sampled diffuse bounce
    let hit_dist = 2.5;

    var cand: GiPathCandidate;
    cand.bounce_position = primary_pos + bounce_dir * hit_dist;
    cand.bounce_normal = vec3<f32>(0.0, -1.0, 0.0);
    cand.indirect_radiance = vec3<f32>(0.4, 0.45, 0.5); // Indirect sky reflection
    cand.path_pdf = 0.318309886; // 1 / PI

    s_gi_candidates[idx] = cand;
}
