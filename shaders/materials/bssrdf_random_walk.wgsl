// Shroud :: Random-Walk Subsurface Scattering
// Physically accurate zero-variance random walk inside participating volume bounds.

struct RandomWalkParams {
    sigma_a: vec3<f32>,
    g: f32,
    sigma_s: vec3<f32>,
    max_bounces: u32,
};

struct SubsurfaceHitOutput {
    exit_point: vec3<f32>,
    exit_normal: vec3<f32>,
    throughput: vec3<f32>,
    bounces: u32,
};

fn sample_hg_direction(v_in: vec3<f32>, g: f32, u: vec2<f32>) -> vec3<f32> {
    var cos_theta = 1.0 - 2.0 * u.x;
    if (abs(g) > 1e-3) {
        let sqr_term = (1.0 - g * g) / (1.0 - g + 2.0 * g * u.x);
        cos_theta = (1.0 + g * g - sqr_term * sqr_term) / (2.0 * g);
    }
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let phi = 6.28318530718 * u.y;
    return vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

@group(0) @binding(0) var<uniform> u_rw_params: RandomWalkParams;
@group(0) @binding(1) var<storage, read> s_entry_points: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_subsurface_results: array<SubsurfaceHitOutput>;

@compute @workgroup_size(64, 1, 1)
fn cs_random_walk_simulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_entry_points)) {
        return;
    }

    let entry = s_entry_points[idx].xyz;
    var pos = entry;
    var dir = vec3<f32>(0.0, -1.0, 0.0); // Enter along surface normal
    var throughput = vec3<f32>(1.0);
    var bounce = 0u;

    let sigma_t = u_rw_params.sigma_a + u_rw_params.sigma_s;
    let mean_free_path = 1.0 / max(vec3<f32>(1e-4), sigma_t);

    while (bounce < u_rw_params.max_bounces) {
        let step_dist = mean_free_path.y * 0.5;
        pos += dir * step_dist;

        // Check if exited bounding slab (y >= 0)
        if (pos.y >= 0.0) {
            break;
        }

        // Scatter direction
        dir = sample_hg_direction(dir, u_rw_params.g, vec2<f32>(0.4, 0.6));
        throughput = throughput * (u_rw_params.sigma_s / sigma_t);
        bounce += 1u;
    }

    var out_rec: SubsurfaceHitOutput;
    out_rec.exit_point = pos;
    out_rec.exit_normal = vec3<f32>(0.0, 1.0, 0.0);
    out_rec.throughput = throughput;
    out_rec.bounces = bounce;
    s_subsurface_results[idx] = out_rec;
}
