// Shroud :: SVGF Temporal Accumulation Pass (Schied et al. 2017)
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
