// Shroud :: Multi-Layer Clearcoat & Metallic Flake Vehicle Coating
// Clearcoat Fresnel reflection, absorption depth attenuation, and base metallicity.

struct LayeredCoatingParams {
    coat_roughness: f32,
    coat_ior: f32,
    coat_thickness_um: f32,
    padding0: f32,
    coat_absorption_color: vec3<f32>,
    padding1: f32,
    base_metallic: f32,
    base_roughness: f32,
    padding2: vec2<f32>,
};

fn beer_lambert_attenuation(absorption: vec3<f32>, distance_um: f32) -> vec3<f32> {
    return exp(-absorption * distance_um * 0.001);
}

@group(0) @binding(0) var<uniform> u_coating: LayeredCoatingParams;
@group(0) @binding(1) var<storage, read> s_layer_rays: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_coating_composite: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_layered_coating_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_layer_rays)) {
        return;
    }

    let cos_t = s_layer_rays[idx].w;
    let n0 = 1.0;
    let n1 = u_coating.coat_ior;

    let f0 = pow((n1 - n0) / (n1 + n0), 2.0);
    let f_coat = f0 + (1.0 - f0) * pow(clamp(1.0 - cos_t, 0.0, 1.0), 5.0);

    let path_length = u_coating.coat_thickness_um / max(1e-4, cos_t);
    let trans = beer_lambert_attenuation(u_coating.coat_absorption_color, path_length);

    let base_contrib = vec3<f32>(0.8, 0.1, 0.1) * (1.0 - f_coat) * trans;
    let composite = vec3<f32>(f_coat) + base_contrib;

    s_coating_composite[idx] = vec4<f32>(composite, 1.0);
}
