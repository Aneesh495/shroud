// Shroud :: Linear Bounding Volume Hierarchy (LBVH) Morton Code Generation
// Computes 30-bit Morton codes via bit interleaving of normalized 3D coordinates.

struct BvhPrimitiveInput {
    v0: vec3<f32>,
    prim_id: u32,
    v1: vec3<f32>,
    material_id: u32,
    v2: vec3<f32>,
    padding: u32,
};

struct MortonKeyOutput {
    morton_code: u32,
    primitive_id: u32,
};

struct SceneBoundsUniforms {
    scene_min: vec3<f32>,
    num_primitives: u32,
    scene_max: vec3<f32>,
    padding: u32,
};

@group(0) @binding(0) var<uniform> u_scene_bounds: SceneBoundsUniforms;
@group(0) @binding(1) var<storage, read> s_primitives: array<BvhPrimitiveInput>;
@group(0) @binding(2) var<storage, read_write> s_morton_keys: array<MortonKeyOutput>;

fn expand_bits_10(v_in: u32) -> u32 {
    var v = v_in & 0x000003FFu;
    v = (v ^ (v << 16u)) & 0x030000FFu;
    v = (v ^ (v <<  8u)) & 0x0300F00Fu;
    v = (v ^ (v <<  4u)) & 0x030C30C3u;
    v = (v ^ (v <<  2u)) & 0x09249249u;
    return v;
}

fn compute_morton_30(norm_coord: vec3<f32>) -> u32 {
    let clamped = clamp(norm_coord, vec3<f32>(0.0), vec3<f32>(1.0));
    let x = u32(clamped.x * 1023.0);
    let y = u32(clamped.y * 1023.0);
    let z = u32(clamped.z * 1023.0);
    return expand_bits_10(x) | (expand_bits_10(y) << 1u) | (expand_bits_10(z) << 2u);
}

@compute @workgroup_size(128, 1, 1)
fn cs_generate_morton_codes(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let prim_idx = global_id.x;
    if (prim_idx >= u_scene_bounds.num_primitives) {
        return;
    }

    let prim = s_primitives[prim_idx];
    let centroid = (prim.v0 + prim.v1 + prim.v2) * 0.33333333333;

    let extent = u_scene_bounds.scene_max - u_scene_bounds.scene_min;
    var norm_pos = vec3<f32>(0.5, 0.5, 0.5);
    if (extent.x > 1e-6 && extent.y > 1e-6 && extent.z > 1e-6) {
        norm_pos = (centroid - u_scene_bounds.scene_min) / extent;
    }

    let code = compute_morton_30(norm_pos);

    var key_out: MortonKeyOutput;
    key_out.morton_code = code;
    key_out.primitive_id = prim_idx;

    s_morton_keys[prim_idx] = key_out;
}
