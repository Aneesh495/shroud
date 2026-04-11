// Shroud :: Visible Normal Distribution (VNDF) GGX Importance Sampling
// Implements Heitz 2018 bounded VNDF sampling algorithm.

const PI: f32 = 3.14159265358979323846;

fn sample_ggx_vndf(v_in: vec3<f32>, alpha_x: f32, alpha_y: f32, u: vec2<f32>) -> vec3<f32> {
    // Transform view direction to hemisphere configuration
    let v_h = normalize(vec3<f32>(alpha_x * v_in.x, alpha_y * v_in.y, v_in.z));

    // Orthonormal basis around v_h
    let lensq = v_h.x * v_h.x + v_h.y * v_h.y;
    let t1 = select(vec3<f32>(-v_h.y, v_h.x, 0.0) * inverseSqrt(lensq), vec3<f32>(1.0, 0.0, 0.0), lensq < 1e-8);
    let t2 = cross(v_h, t1);

    // Parameterized disk sampling
    let r = sqrt(u.x);
    let phi = 2.0 * PI * u.y;
    let p1 = r * cos(phi);
    let p2 = r * sin(phi);
    let s = 0.5 * (1.0 + v_h.z);
    let p2_clamped = (1.0 - s) * sqrt(max(0.0, 1.0 - p1 * p1)) + s * p2;

    // Unproject to hemisphere
    let n_h = p1 * t1 + p2_clamped * t2 + sqrt(max(0.0, 1.0 - p1 * p1 - p2_clamped * p2_clamped)) * v_h;

    // Transform back to ellipsoid configuration
    return normalize(vec3<f32>(alpha_x * n_h.x, alpha_y * n_h.y, max(0.0, n_h.z)));
}

@group(0) @binding(0) var<storage, read> s_view_dirs: array<vec4<f32>>; // view, roughness
@group(0) @binding(1) var<storage, read> s_random_u: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> s_sampled_normals: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_sample_vndf_normals(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_sampled_normals)) {
        return;
    }
    let v = s_view_dirs[idx].xyz;
    let rough = max(0.001, s_view_dirs[idx].w);
    let u = s_random_u[idx];
    let n_sample = sample_ggx_vndf(v, rough, rough, u);
    s_sampled_normals[idx] = vec4<f32>(n_sample, 1.0);
}
