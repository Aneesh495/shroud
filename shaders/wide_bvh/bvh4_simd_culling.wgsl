// Shroud :: 4-Wide BVH (BVH4) Quad-Box Traversal Kernel
// Intersects 4 bounding boxes simultaneously to maximize GPU memory bandwidth.

struct Bvh4Node {
    bbox_min_x: vec4<f32>,
    bbox_max_x: vec4<f32>,
    bbox_min_y: vec4<f32>,
    bbox_max_y: vec4<f32>,
    bbox_min_z: vec4<f32>,
    bbox_max_z: vec4<f32>,
    child_indices: vec4<u32>,
};

struct Ray4Query {
    origin: vec3<f32>,
    inv_dir: vec3<f32>,
    t_min: f32,
    t_max: f32,
};

fn intersect_bvh4_children(ray: Ray4Query, node: Bvh4Node) -> vec4<f32> {
    let t0_x = (node.bbox_min_x - ray.origin.x) * ray.inv_dir.x;
    let t1_x = (node.bbox_max_x - ray.origin.x) * ray.inv_dir.x;
    let tmin_x = min(t0_x, t1_x);
    let tmax_x = max(t0_x, t1_x);

    let t0_y = (node.bbox_min_y - ray.origin.y) * ray.inv_dir.y;
    let t1_y = (node.bbox_max_y - ray.origin.y) * ray.inv_dir.y;
    let tmin_y = min(t0_y, t1_y);
    let tmax_y = max(t0_y, t1_y);

    let t0_z = (node.bbox_min_z - ray.origin.z) * ray.inv_dir.z;
    let t1_z = (node.bbox_max_z - ray.origin.z) * ray.inv_dir.z;
    let tmin_z = min(t0_z, t1_z);
    let tmax_z = max(t0_z, t1_z);

    let t_enter = max(max(tmin_x, tmin_y), tmin_z);
    let t_exit = min(min(tmax_x, tmax_y), tmax_z);

    return select(vec4<f32>(-1.0), max(vec4<f32>(ray.t_min), t_enter), t_exit >= max(vec4<f32>(0.0), t_enter) && t_enter <= vec4<f32>(ray.t_max));
}

@group(0) @binding(0) var<storage, read> s_rays: array<Ray4Query>;
@group(0) @binding(1) var<storage, read> s_nodes: array<Bvh4Node>;
@group(0) @binding(2) var<storage, read_write> s_child_distances: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_traverse_bvh4(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_rays) || arrayLength(&s_nodes) == 0u) {
        return;
    }
    let r = s_rays[idx];
    let node = s_nodes[0];
    s_child_distances[idx] = intersect_bvh4_children(r, node);
}
