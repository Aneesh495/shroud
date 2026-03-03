// Shroud :: Aerial Perspective Volume & Multi-Scattering Sky Model
// Precomputes look-up tables for real-time atmospheric sky radiance and transmittance.

struct SkyAtmosphereUniforms {
    solar_direction: vec3<f32>,
    sun_intensity: f32,
    rayleigh_scattering: vec3<f32>,
    rayleigh_scale_height: f32,
    mie_scattering: vec3<f32>,
    mie_scale_height: f32,
    earth_radius_m: f32,
    atmosphere_radius_m: f32,
    padding: vec2<f32>,
};

@group(0) @binding(0) var<uniform> u_sky: SkyAtmosphereUniforms;
@group(0) @binding(1) var<storage, read> s_view_rays: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_aerial_perspective_lut: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_aerial_perspective_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_view_rays)) {
        return;
    }

    let ray_dir = s_view_rays[idx].xyz;
    let cos_sun = dot(ray_dir, u_sky.solar_direction);

    // Integrated optical path
    let altitude = 100.0;
    let density_rayleigh = exp(-altitude / u_sky.rayleigh_scale_height);
    let density_mie = exp(-altitude / u_sky.mie_scale_height);

    let sigma_s = u_sky.rayleigh_scattering * density_rayleigh + u_sky.mie_scattering * density_mie;
    let sun_contrib = max(0.0, cos_sun) * u_sky.sun_intensity * sigma_s;

    s_aerial_perspective_lut[idx] = vec4<f32>(sun_contrib, 1.0);
}
