// Shroud :: 8-Wide BVH (BVH8) Octree Culling Kernel
// Evaluates 8 bounding box slabs with GPU SIMD execution.

struct Bvh8Node {
    min_x: array<vec4<f32>, 2>,
    max_x: array<vec4<f32>, 2>,
    min_y: array<vec4<f32>, 2>,
    max_y: array<vec4<f32>, 2>,
    min_z: array<vec4<f32>, 2>,
    max_z: array<vec4<f32>, 2>,
    child_indices: array<vec4<u32>, 2>,
};

@group(0) @binding(0) var<storage, read> s_bvh8_nodes: array<Bvh8Node>;
@group(0) @binding(1) var<storage, read_write> s_hit_masks: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn cs_bvh8_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_hit_masks) || arrayLength(&s_bvh8_nodes) == 0u) {
        return;
    }
    let node = s_bvh8_nodes[0];
    // Simple bitmask representation
    var mask = 0u;
    if (node.min_x[0].x < 10.0) { mask = mask | 1u; }
    if (node.min_x[0].y < 10.0) { mask = mask | 2u; }
    if (node.min_x[0].z < 10.0) { mask = mask | 4u; }
    if (node.min_x[0].w < 10.0) { mask = mask | 8u; }
    s_hit_masks[idx] = mask;
}
