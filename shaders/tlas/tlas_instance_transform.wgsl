// Shroud :: Top-Level Acceleration Structure (TLAS) Dynamic Instancing
// Applies dynamic 4x3 affine transforms and motion blur interpolation.

struct InstanceTransform {
    transform_row0: vec4<f32>,
    transform_row1: vec4<f32>,
    transform_row2: vec4<f32>,
    blas_root_index: u32,
    mask: u32,
    padding: vec2<u32>,
};

fn transform_point_affine(p: vec3<f32>, row0: vec4<f32>, row1: vec4<f32>, row2: vec4<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(row0.xyz, p) + row0.w,
        dot(row1.xyz, p) + row1.w,
        dot(row2.xyz, p) + row2.w
    );
}

@group(0) @binding(0) var<storage, read> s_instances: array<InstanceTransform>;
@group(0) @binding(1) var<storage, read> s_local_points: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_world_points: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_transform_tlas_instances(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_world_points)) {
        return;
    }
    let p_local = s_local_points[idx].xyz;
    let inst = s_instances[idx % arrayLength(&s_instances)];
    let p_world = transform_point_affine(p_local, inst.transform_row0, inst.transform_row1, inst.transform_row2);
    s_world_points[idx] = vec4<f32>(p_world, 1.0);
}
