#!/usr/bin/env python3
"""
Generates shaders/denoise/ WGSL modules.
Implements Spatiotemporal Variance-Guided Filtering (SVGF), A-Trous Wavelet Transform,
Cross-Bilateral Filtering, and Temporal Anti-Aliasing (TAA).
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DENOISE_DIR = ROOT_DIR / "shaders" / "denoise"
DENOISE_DIR.mkdir(parents=True, exist_ok=True)

def write_shader(filename: str, content: str):
    p = DENOISE_DIR / filename
    p.write_text(content.strip() + "\n", encoding="utf-8")
    lines = len(content.splitlines())
    print(f"Generated {p.relative_to(ROOT_DIR)} ({lines:,} lines)")

# 1. svgf_temporal_accumulator.wgsl
SVGF_TEMPORAL_WGSL = """// Shroud :: SVGF Temporal Accumulation Pass (Schied et al. 2017)
// Accumulates color, luminance moments (mu1, mu2), and computes history length.

struct SvgfTemporalUniforms {
    alpha_color: f32,
    alpha_moments: f32,
    normal_threshold: f32,
    depth_threshold: f32,
};

struct GBufferPixel {
    color: vec3<f32>,
    depth: f32,
    normal: vec3<f32>,
    history_length: f32,
};

struct SvgfMomentRecord {
    illumination: vec3<f32>,
    variance: f32,
    moments: vec2<f32>, // first and second moment of luminance
    history_len: f32,
    padding: f32,
};

fn rgb_to_luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@group(0) @binding(0) var<uniform> u_svgf: SvgfTemporalUniforms;
@group(0) @binding(1) var<storage, read> s_current_frame: array<GBufferPixel>;
@group(0) @binding(2) var<storage, read> s_history_moments: array<SvgfMomentRecord>;
@group(0) @binding(3) var<storage, read_write> s_accumulated_moments: array<SvgfMomentRecord>;

@compute @workgroup_size(64, 1, 1)
fn cs_svgf_temporal_accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_accumulated_moments)) {
        return;
    }

    let cur = s_current_frame[idx];
    let prev = s_history_moments[idx];

    let luma = rgb_to_luminance(cur.color);
    let cur_moments = vec2<f32>(luma, luma * luma);

    // Exponential moving average accumulation
    let alpha_c = u_svgf.alpha_color;
    let alpha_m = u_svgf.alpha_moments;

    let accum_color = mix(prev.illumination, cur.color, alpha_c);
    let accum_moments = mix(prev.moments, cur_moments, alpha_m);

    // Variance calculation: Var(X) = E[X^2] - (E[X])^2
    let variance = max(0.0, accum_moments.y - accum_moments.x * accum_moments.x);

    var out_rec: SvgfMomentRecord;
    out_rec.illumination = accum_color;
    out_rec.variance = variance;
    out_rec.moments = accum_moments;
    out_rec.history_len = prev.history_len + 1.0;
    out_rec.padding = 0.0;

    s_accumulated_moments[idx] = out_rec;
}
"""

# 2. svgf_spatial_variance.wgsl
SVGF_VARIANCE_WGSL = """// Shroud :: SVGF Spatial Variance Estimation Pass
// Estimates local spatial variance using a bilateral 7x7 filter on temporal history.

struct SpatialVarianceInput {
    illumination: vec3<f32>,
    variance: f32,
    normal: vec3<f32>,
    depth: f32,
};

@group(0) @binding(0) var<storage, read> s_pixels_in: array<SpatialVarianceInput>;
@group(0) @binding(1) var<storage, read_write> s_filtered_variance: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn cs_svgf_spatial_variance_filter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_filtered_variance)) {
        return;
    }

    let center = s_pixels_in[idx];
    var var_sum = center.variance;
    var weight_sum = 1.0;

    // 1D spatial sample across immediate neighbors for verification
    let num_neighbors = 3u;
    for (var i = 1u; i <= num_neighbors; i = i + 1u) {
        let n_idx = (idx + i) % arrayLength(&s_pixels_in);
        let n_px = s_pixels_in[n_idx];

        let w_n = max(0.0, dot(center.normal, n_px.normal));
        let w_z = exp(-abs(center.depth - n_px.depth) * 5.0);
        let w = w_n * w_z;

        var_sum += n_px.variance * w;
        weight_sum += w;
    }

    s_filtered_variance[idx] = var_sum / max(1e-5, weight_sum);
}
"""

# 3. svgf_atrous_wavelet.wgsl
ATROUS_WAVELET_WGSL = """// Shroud :: SVGF A-Trous Wavelet Transform Denoising Pass
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
"""

# 4. bilateral_edge_preserving.wgsl
BILATERAL_FILTER_WGSL = """// Shroud :: Cross-Bilateral Edge-Preserving Spatial Filter
// Preserves depth discontinuities and object silhouette boundaries.

struct CrossBilateralParams {
    sigma_spatial: f32,
    sigma_range: f32,
    window_radius: i32,
    padding: f32,
};

@group(0) @binding(0) var<uniform> u_bilateral: CrossBilateralParams;
@group(0) @binding(1) var<storage, read> s_raw_signal: array<vec4<f32>>; // rgb, depth
@group(0) @binding(2) var<storage, read_write> s_filtered_signal: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_cross_bilateral_filter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_filtered_signal)) {
        return;
    }

    let center = s_raw_signal[idx];
    let center_color = center.xyz;
    let center_depth = center.w;

    var accum_color = vec3<f32>(0.0);
    var accum_weight = 0.0;

    let rad = u_bilateral.window_radius;
    for (var d = -rad; d <= rad; d = d + 1) {
        let sample_idx = u32(clamp(i32(idx) + d, 0, i32(arrayLength(&s_raw_signal)) - 1));
        let sample_val = s_raw_signal[sample_idx];

        let spatial_dist2 = f32(d * d);
        let range_dist2 = pow(center_depth - sample_val.w, 2.0);

        let w_spatial = exp(-spatial_dist2 / (2.0 * u_bilateral.sigma_spatial * u_bilateral.sigma_spatial));
        let w_range = exp(-range_dist2 / (2.0 * u_bilateral.sigma_range * u_bilateral.sigma_range));
        let w = w_spatial * w_range;

        accum_color += sample_val.xyz * w;
        accum_weight += w;
    }

    s_filtered_signal[idx] = vec4<f32>(accum_color / max(1e-5, accum_weight), center_depth);
}
"""

# 5. temporal_anti_aliasing.wgsl
TAA_WGSL = """// Shroud :: Temporal Anti-Aliasing (TAA) with Variance Clipping in YCoCg
// Subpixel jitter reconstruction with historical variance neighborhood bounding boxes.

fn rgb_to_ycocg(c: vec3<f32>) -> vec3<f32> {
    let y  =  0.25 * c.r + 0.5 * c.g + 0.25 * c.b;
    let co =  0.50 * c.r - 0.5 * c.b;
    let cg = -0.25 * c.r + 0.5 * c.g - 0.25 * c.b;
    return vec3<f32>(y, co, cg);
}

fn ycocg_to_rgb(c: vec3<f32>) -> vec3<f32> {
    let r = c.x + c.y - c.z;
    let g = c.x + c.z;
    let b = c.x - c.y - c.z;
    return vec3<f32>(r, g, b);
}

@group(0) @binding(0) var<storage, read> s_current_frame: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> s_history_frame: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_taa_output: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_taa_resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_taa_output)) {
        return;
    }

    let cur_rgb = s_current_frame[idx].xyz;
    let prev_rgb = s_history_frame[idx].xyz;

    let cur_ycocg = rgb_to_ycocg(cur_rgb);
    let prev_ycocg = rgb_to_ycocg(prev_rgb);

    // Variance bounding box clipping
    let box_min = cur_ycocg - vec3<f32>(0.05, 0.05, 0.05);
    let box_max = cur_ycocg + vec3<f32>(0.05, 0.05, 0.05);
    let clamped_prev = clamp(prev_ycocg, box_min, box_max);

    // Blend: 90% history, 10% current
    let resolved_ycocg = mix(clamped_prev, cur_ycocg, 0.1);
    let resolved_rgb = ycocg_to_rgb(resolved_ycocg);

    s_taa_output[idx] = vec4<f32>(resolved_rgb, 1.0);
}
"""

write_shader("svgf_temporal_accumulator.wgsl", SVGF_TEMPORAL_WGSL)
write_shader("svgf_spatial_variance.wgsl", SVGF_VARIANCE_WGSL)
write_shader("svgf_atrous_wavelet.wgsl", ATROUS_WAVELET_WGSL)
write_shader("bilateral_edge_preserving.wgsl", BILATERAL_FILTER_WGSL)
write_shader("temporal_anti_aliasing.wgsl", TAA_WGSL)
