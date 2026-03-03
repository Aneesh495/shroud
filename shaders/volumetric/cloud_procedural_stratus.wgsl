// Shroud :: Low-Altitude Radiation Fog & Stratus Cloud Slab
// Models altitude-bounded temperature inversion and maritime fog banks.

struct StratusFogParams {
    fog_base_m: f32,
    fog_top_m: f32,
    max_extinction: f32,
    droplet_effective_radius_um: f32,
};

fn evaluate_stratus_density(altitude_m: f32, params: StratusFogParams) -> f32 {
    if (altitude_m < params.fog_base_m || altitude_m > params.fog_top_m) {
        return 0.0;
    }

    let h = params.fog_top_m - params.fog_base_m;
    let norm_h = (altitude_m - params.fog_base_m) / max(1.0, h);

    // Parabolic liquid water content (LWC) profile
    let profile = 4.0 * norm_h * (1.0 - norm_h);
    return profile * params.max_extinction;
}

@group(0) @binding(0) var<uniform> u_stratus: StratusFogParams;
@group(0) @binding(1) var<storage, read> s_altitudes: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_extinction_profiles: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn cs_stratus_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_altitudes)) {
        return;
    }
    let alt = s_altitudes[idx];
    s_extinction_profiles[idx] = evaluate_stratus_density(alt, u_stratus);
}
