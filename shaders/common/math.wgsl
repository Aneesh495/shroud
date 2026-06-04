// Shroud :: Geometric & Orthonormal Basis Mathematical Utilities
// Module: shaders/common/math.wgsl
// Duff et al. (2017) branchless orthonormal basis and safe trigonometric helpers.

struct FrameBasis {
    tangent: vec4<f32>,
    bitangent: vec4<f32>,
    normal: vec4<f32>,
};

fn build_onb_duff(n_in: vec3<f32>) -> FrameBasis {
    let n = normalize(n_in);
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let b1 = vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let b2 = vec3<f32>(b, s + n.y * n.y * a, -n.y);
    return FrameBasis(vec4<f32>(b1, 0.0), vec4<f32>(b2, 0.0), vec4<f32>(n, 0.0));
}

@group(0) @binding(0) var<storage, read> s_normals_in: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> s_frames_out: array<FrameBasis>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&s_normals_in) && idx < arrayLength(&s_frames_out)) {
        s_frames_out[idx] = build_onb_duff(s_normals_in[idx].xyz);
    }
}
