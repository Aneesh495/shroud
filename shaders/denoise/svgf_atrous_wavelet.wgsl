// Shroud :: SVGF A-Trous Wavelet Transform Denoising Pass
// Multi-scale 5x5 kernel with dyadic step size 2^k and edge-stopping weights.

struct AtrousParams {
    step_size: i32,
    phi_color: f32,
    phi_normal: f32,
    phi_depth: f32,
};

struct AtrousPixel {
    color: vec3<f32>,
    variance: f32,
    normal: vec3<f32>,
    depth: f32,
};

// 1D Gaussian kernel weights: 1/16, 4/16, 6/16, 4/16, 1/16
const KERNEL_WEIGHTS: array<f32, 5> = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);

@group(0) @binding(0) var<uniform> u_atrous: AtrousParams;
@group(0) @binding(1) var<storage, read> s_atrous_in: array<AtrousPixel>;
@group(0) @binding(2) var<storage, read_write> s_atrous_out: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_atrous_wavelet_filter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_atrous_out)) {
        return;
    }

    let center = s_atrous_in[idx];
    var color_sum = center.color * KERNEL_WEIGHTS[2];
    var weight_sum = KERNEL_WEIGHTS[2];

    let eps_var = max(1e-4, center.variance);

    for (var r = -2; r <= 2; r = r + 1) {
        if (r == 0) {
            continue;
        }

        let n_idx = u32(clamp(i32(idx) + r * u_atrous.step_size, 0, i32(arrayLength(&s_atrous_in)) - 1));
        let neighbor = s_atrous_in[n_idx];

        let kernel_w = KERNEL_WEIGHTS[r + 2];

        // Edge-stopping weights
        let diff_c = center.color - neighbor.color;
        let dist_c2 = dot(diff_c, diff_c);
        let w_c = exp(-dist_c2 / (u_atrous.phi_color * eps_var));

        let w_n = pow(max(0.0, dot(center.normal, neighbor.normal)), u_atrous.phi_normal);
        let w_z = exp(-abs(center.depth - neighbor.depth) / (u_atrous.phi_depth + 1e-4));

        let w = kernel_w * w_c * w_n * w_z;
        color_sum += neighbor.color * w;
        weight_sum += w;
    }

    s_atrous_out[idx] = vec4<f32>(color_sum / max(1e-5, weight_sum), 1.0);
}
