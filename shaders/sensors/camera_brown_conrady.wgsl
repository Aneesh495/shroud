// Shroud :: Brown-Conrady 5-Parameter Optical Distortion Model
// Applies radial (k1, k2, k3) and tangential (p1, p2) lens distortion.

struct DistortionParams {
    k1: f32,
    k2: f32,
    p1: f32,
    p2: f32,
    k3: f32,
    padding: vec3<f32>,
};

fn apply_distortion(p: vec2<f32>, d: DistortionParams) -> vec2<f32> {
    let r2 = dot(p, p);
    let r4 = r2 * r2;
    let r6 = r4 * r2;

    let radial = 1.0 + d.k1 * r2 + d.k2 * r4 + d.k3 * r6;
    let dx_tangential = 2.0 * d.p1 * p.x * p.y + d.p2 * (r2 + 2.0 * p.x * p.x);
    let dy_tangential = d.p1 * (r2 + 2.0 * p.y * p.y) + 2.0 * d.p2 * p.x * p.y;

    return p * radial + vec2<f32>(dx_tangential, dy_tangential);
}

@group(0) @binding(0) var<uniform> u_dist: DistortionParams;
@group(0) @binding(1) var<storage, read> s_undistorted_uvs: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> s_distorted_uvs: array<vec2<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_distort_coordinates(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_undistorted_uvs)) {
        return;
    }
    let p = s_undistorted_uvs[idx];
    s_distorted_uvs[idx] = apply_distortion(p, u_dist);
}
