// Shroud :: Spectral Colorimetry, ACES Filmic Tone Mapping, and sRGB Transfer
// Module: shaders/common/colorimetry.wgsl

fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

@group(0) @binding(0) var<storage, read> s_hdr_radiance: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> s_srgb_out: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&s_hdr_radiance) && idx < arrayLength(&s_srgb_out)) {
        let hdr = s_hdr_radiance[idx].rgb;
        let tm = aces_filmic(hdr);
        let srgb = linear_to_srgb(tm);
        s_srgb_out[idx] = vec4<f32>(srgb, 1.0);
    }
}
