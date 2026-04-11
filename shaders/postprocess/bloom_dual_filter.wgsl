// Shroud :: Dual-Filter Optical Glare & Bloom Convolution
// Efficient multi-pass downsampling and upsampling pyramid blur.

struct BloomUniforms {
    threshold: f32,
    soft_knee: f32,
    intensity: f32,
    padding: f32,
};

@group(0) @binding(0) var<uniform> u_bloom: BloomUniforms;
@group(0) @binding(1) var<storage, read> s_hdr_color_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_bloom_pyramid_out: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_bloom_downsample_filter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_bloom_pyramid_out)) {
        return;
    }
    let c = s_hdr_color_in[idx].xyz;
    let luma = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    let excess = max(0.0, luma - u_bloom.threshold);
    let factor = excess / max(1e-4, luma);
    s_bloom_pyramid_out[idx] = vec4<f32>(c * factor * u_bloom.intensity, 1.0);
}
