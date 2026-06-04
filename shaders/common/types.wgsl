// Shroud :: Common GPU Memory Layouts (std140/std430 Aligned)
// Module: shaders/common/types.wgsl
// Canonical GPU structs matching Rust `bytemuck::Pod` definitions in shroud-core.

struct RayGpu {
    origin: vec4<f32>,
    direction: vec4<f32>,
};

struct AabbGpu {
    min_pt: vec4<f32>,
    max_pt: vec4<f32>,
};

struct BvhNodeGpu {
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    left_or_first: u32,
    right_or_second: u32,
    prim_count: u32,
    pad: u32,
};

struct MediaPropertiesGpu {
    sigma_a: f32,
    sigma_s: f32,
    asymmetry_g: f32,
    pad: f32,
};

@group(0) @binding(0) var<storage, read> s_rays: array<RayGpu>;
@group(0) @binding(1) var<storage, read_write> s_bounds: array<AabbGpu>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&s_rays) && idx < arrayLength(&s_bounds)) {
        let r = s_rays[idx];
        let p_far = r.origin.xyz + normalize(r.direction.xyz) * 100.0;
        let lo = min(r.origin.xyz, p_far);
        let hi = max(r.origin.xyz, p_far);
        s_bounds[idx] = AabbGpu(vec4<f32>(lo, 1.0), vec4<f32>(hi, 1.0));
    }
}
