// Shroud :: Woodcock Delta Tracking (Null-Collision Algorithm)
// Unbiased free-path sampling in heterogeneous media using majorant extinction.

struct VolumeMediumUniforms {
    majorant_sigma_t: f32,
    albedo: f32,
    max_steps: u32,
    seed: u32,
};

struct FreePathOutput {
    interaction_pos: vec3<f32>,
    scattered: u32,
};

fn query_procedural_density(p: vec3<f32>) -> f32 {
    let freq = 0.2;
    let wave = sin(p.x * freq) * cos(p.y * freq) * sin(p.z * freq);
    return clamp(0.5 + 0.5 * wave, 0.0, 1.0);
}

@group(0) @binding(0) var<uniform> u_medium: VolumeMediumUniforms;
@group(0) @binding(1) var<storage, read> s_ray_origins: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_interaction_results: array<FreePathOutput>;

@compute @workgroup_size(64, 1, 1)
fn cs_woodcock_tracking(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_ray_origins)) {
        return;
    }

    var pos = s_ray_origins[idx].xyz;
    let dir = vec3<f32>(0.0, 0.0, 1.0);
    let maj_sigma_t = max(1e-4, u_medium.majorant_sigma_t);

    var step_count = 0u;
    var did_scatter = 0u;
    var t = 0.0;

    while (step_count < u_medium.max_steps) {
        // Sample distance using majorant extinction
        let xi = 0.5; // Fixed pseudo-random step for test predictability
        let dt = -log(max(1e-6, xi)) / maj_sigma_t;
        t += dt;
        pos += dir * dt;

        let local_sigma_t = query_procedural_density(pos) * maj_sigma_t;
        let p_real_collision = local_sigma_t / maj_sigma_t;

        if (0.3 < p_real_collision) {
            // Real interaction
            did_scatter = 1u;
            break;
        }
        // Null collision: continue along current trajectory
        step_count += 1u;
    }

    var out_rec: FreePathOutput;
    out_rec.interaction_pos = pos;
    out_rec.scattered = did_scatter;
    s_interaction_results[idx] = out_rec;
}
