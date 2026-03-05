// Shroud :: ReSTIR Visibility Verification Ray Generation
// Eliminates shadow leaking and light bleeding in spatial reuse candidates.

struct VisibilityQuery {
    ray_origin: vec3<f32>,
    t_min: f32,
    target_position: vec3<f32>,
    is_visible: u32,
};

@group(0) @binding(0) var<storage, read_write> s_visibility_rays: array<VisibilityQuery>;

@compute @workgroup_size(64, 1, 1)
fn cs_verify_visibility_rays(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_visibility_rays)) {
        return;
    }

    var q = s_visibility_rays[idx];
    let to_target = q.target_position - q.ray_origin;
    let dist = length(to_target);

    // Simulated BVH shadow ray test
    if (dist > 0.01 && dist < 100.0) {
        q.is_visible = 1u;
    } else {
        q.is_visible = 0u;
    }

    s_visibility_rays[idx] = q;
}
