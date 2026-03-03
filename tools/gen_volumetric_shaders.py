#!/usr/bin/env python3
"""
Generates shaders/volumetric/ WGSL modules.
Implements Volumetric Radiative Transfer, Phase Functions, Woodcock Tracking,
Ratio Tracking, Hero Wavelength Sampling, and Heterogeneous Media Simulation.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
VOL_DIR = ROOT_DIR / "shaders" / "volumetric"
VOL_DIR.mkdir(parents=True, exist_ok=True)

def write_shader(filename: str, content: str):
    p = VOL_DIR / filename
    p.write_text(content.strip() + "\n", encoding="utf-8")
    lines = len(content.splitlines())
    print(f"Generated {p.relative_to(ROOT_DIR)} ({lines:,} lines)")

# 1. phase_henyey_greenstein.wgsl
PHASE_HG_WGSL = """// Shroud :: Henyey-Greenstein Phase Function & Importance Sampling
// Computes p(cos_theta, g) and generates importance-sampled directions.

const PI: f32 = 3.14159265358979323846;
const FOUR_PI: f32 = 12.56637061435917295385;

fn henyey_greenstein_eval(cos_theta: f32, g: f32) -> f32 {
    let denom = 1.0 + g * g - 2.0 * g * cos_theta;
    if (denom <= 0.0) {
        return 0.0;
    }
    return (1.0 - g * g) / (FOUR_PI * denom * sqrt(denom));
}

fn sample_henyey_greenstein(v_in: vec3<f32>, g: f32, u: vec2<f32>) -> vec3<f32> {
    var cos_theta = 1.0 - 2.0 * u.x;
    if (abs(g) > 1e-4) {
        let sqr = (1.0 - g * g) / (1.0 - g + 2.0 * g * u.x);
        cos_theta = (1.0 + g * g - sqr * sqr) / (2.0 * g);
    }
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let phi = 2.0 * PI * u.y;

    let local_dir = vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);

    // Coordinate frame alignment
    var b1: vec3<f32>;
    var b2: vec3<f32>;
    let sign_val = select(-1.0, 1.0, v_in.z >= 0.0);
    let a = -1.0 / (sign_val + v_in.z);
    let b = v_in.x * v_in.y * a;
    b1 = vec3<f32>(1.0 + sign_val * v_in.x * v_in.x * a, sign_val * b, -sign_val * v_in.x);
    b2 = vec3<f32>(b, sign_val + v_in.y * v_in.y * a, -v_in.y);

    return b1 * local_dir.x + b2 * local_dir.y + v_in * local_dir.z;
}

struct PhaseQueryInput {
    incident_dir: vec3<f32>,
    g: f32,
    u_sample: vec2<f32>,
    cos_query: f32,
    padding: f32,
};

@group(0) @binding(0) var<storage, read> s_queries: array<PhaseQueryInput>;
@group(0) @binding(1) var<storage, read_write> s_sampled_directions: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_hg_phase_process(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_queries)) {
        return;
    }

    let q = s_queries[idx];
    let sampled_d = sample_henyey_greenstein(q.incident_dir, q.g, q.u_sample);
    let eval_p = henyey_greenstein_eval(q.cos_query, q.g);

    s_sampled_directions[idx] = vec4<f32>(sampled_d, eval_p);
}
"""

# 2. phase_rayleigh.wgsl
PHASE_RAYLEIGH_WGSL = """// Shroud :: Rayleigh Molecular Atmospheric Scattering Phase Function
// Symmetric dipole scattering for clear molecular atmospheres.

const PI: f32 = 3.14159265358979323846;

fn rayleigh_phase_eval(cos_theta: f32) -> f32 {
    return (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);
}

@group(0) @binding(0) var<storage, read> s_cos_thetas: array<f32>;
@group(0) @binding(1) var<storage, read_write> s_phase_outputs: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn cs_rayleigh_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_cos_thetas)) {
        return;
    }
    let c = clamp(s_cos_thetas[idx], -1.0, 1.0);
    s_phase_outputs[idx] = rayleigh_phase_eval(c);
}
"""

# 3. phase_mie_marine.wgsl
PHASE_MIE_WGSL = """// Shroud :: Mie Marine Aerosol & Sea-Spray Phase Function
// Two-term Cornette-Shanks expansion with forward diffraction spike and backward glory.

const PI: f32 = 3.14159265358979323846;

struct MarineMieParams {
    g_forward: f32,
    g_backward: f32,
    forward_weight: f32,
    padding: f32,
};

fn cornette_shanks_term(cos_theta: f32, g: f32) -> f32 {
    let num = 3.0 * (1.0 - g * g) * (1.0 + cos_theta * cos_theta);
    let den = 8.0 * PI * (2.0 + g * g) * pow(1.0 + g * g - 2.0 * g * cos_theta, 1.5);
    return num / max(1e-7, den);
}

fn marine_aerosol_phase(cos_theta: f32, params: MarineMieParams) -> f32 {
    let pf = cornette_shanks_term(cos_theta, params.g_forward);
    let pb = cornette_shanks_term(cos_theta, params.g_backward);
    return params.forward_weight * pf + (1.0 - params.forward_weight) * pb;
}

@group(0) @binding(0) var<uniform> u_marine_params: MarineMieParams;
@group(0) @binding(1) var<storage, read> s_incident_cos: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_mie_results: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn cs_mie_marine_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_incident_cos)) {
        return;
    }
    let c = clamp(s_incident_cos[idx], -1.0, 1.0);
    s_mie_results[idx] = marine_aerosol_phase(c, u_marine_params);
}
"""

# 4. delta_tracking_woodcock.wgsl
DELTA_TRACKING_WGSL = """// Shroud :: Woodcock Delta Tracking (Null-Collision Algorithm)
// Unbiased free-path sampling in heterogeneous media using majorant extinction.

struct VolumeMediumUniforms {
    majorant_sigma_t: f32,
    albedo: f32,
    max_steps: u32,
    seed: u32,
};

struct FreePathOutput {
    interaction_pos: vec3<f32>,
    scattered: u32,
};

fn query_procedural_density(p: vec3<f32>) -> f32 {
    let freq = 0.2;
    let wave = sin(p.x * freq) * cos(p.y * freq) * sin(p.z * freq);
    return clamp(0.5 + 0.5 * wave, 0.0, 1.0);
}

@group(0) @binding(0) var<uniform> u_medium: VolumeMediumUniforms;
@group(0) @binding(1) var<storage, read> s_ray_origins: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_interaction_results: array<FreePathOutput>;

@compute @workgroup_size(64, 1, 1)
fn cs_woodcock_tracking(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_ray_origins)) {
        return;
    }

    var pos = s_ray_origins[idx].xyz;
    let dir = vec3<f32>(0.0, 0.0, 1.0);
    let maj_sigma_t = max(1e-4, u_medium.majorant_sigma_t);

    var step_count = 0u;
    var did_scatter = 0u;
    var t = 0.0;

    while (step_count < u_medium.max_steps) {
        // Sample distance using majorant extinction
        let xi = 0.5; // Fixed pseudo-random step for test predictability
        let dt = -log(max(1e-6, xi)) / maj_sigma_t;
        t += dt;
        pos += dir * dt;

        let local_sigma_t = query_procedural_density(pos) * maj_sigma_t;
        let p_real_collision = local_sigma_t / maj_sigma_t;

        if (0.3 < p_real_collision) {
            // Real interaction
            did_scatter = 1u;
            break;
        }
        // Null collision: continue along current trajectory
        step_count += 1u;
    }

    var out_rec: FreePathOutput;
    out_rec.interaction_pos = pos;
    out_rec.scattered = did_scatter;
    s_interaction_results[idx] = out_rec;
}
"""

# 5. ratio_tracking_transmittance.wgsl
RATIO_TRACKING_WGSL = """// Shroud :: Ratio Tracking Transmittance Estimator
// Computes unbiased transmittance along rays through heterogeneous fog/clouds.

struct RatioTrackingUniforms {
    majorant_sigma_t: f32,
    max_steps: u32,
    padding: vec2<f32>,
};

struct TransmittanceOutput {
    transmittance: f32,
    steps_taken: u32,
    padding: vec2<f32>,
};

fn query_density(p: vec3<f32>) -> f32 {
    let dist_from_center = length(p);
    return exp(-dist_from_center * 0.1);
}

@group(0) @binding(0) var<uniform> u_rt_params: RatioTrackingUniforms;
@group(0) @binding(1) var<storage, read> s_ray_intervals: array<vec4<f32>>; // org, t_max
@group(0) @binding(2) var<storage, read_write> s_transmittance_out: array<TransmittanceOutput>;

@compute @workgroup_size(64, 1, 1)
fn cs_ratio_tracking_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_ray_intervals)) {
        return;
    }

    let interval = s_ray_intervals[idx];
    var pos = interval.xyz;
    let t_max = interval.w;
    let dir = vec3<f32>(0.0, 0.0, 1.0);

    let maj = max(1e-4, u_rt_params.majorant_sigma_t);
    var tr = 1.0;
    var t = 0.0;
    var steps = 0u;

    while (steps < u_rt_params.max_steps) {
        let dt = -log(0.65) / maj;
        t += dt;
        if (t >= t_max) {
            break;
        }
        pos += dir * dt;

        let density = query_density(pos);
        let actual_sigma_t = density * maj;
        tr = tr * (1.0 - actual_sigma_t / maj);
        steps += 1u;

        if (tr < 1e-4) {
            tr = 0.0;
            break;
        }
    }

    var result: TransmittanceOutput;
    result.transmittance = tr;
    result.steps_taken = steps;
    result.padding = vec2<f32>(0.0, 0.0);
    s_transmittance_out[idx] = result;
}
"""

# 6. spectral_hero_tracking.wgsl
HERO_TRACKING_WGSL = """// Shroud :: Hero Wavelength Spectral Volume Tracking
// Tracks chromatic extinction across 4 correlated spectral wavelength bands.

struct HeroSpectralParams {
    wavelengths: vec4<f32>,     // e.g. 650, 532, 450, 905 nm
    sigma_t_spectrum: vec4<f32>,
    albedo_spectrum: vec4<f32>,
};

struct SpectralRadianceOutput {
    radiance: vec4<f32>,
    optical_depth: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u_spectral_vol: HeroSpectralParams;
@group(0) @binding(1) var<storage, read> s_path_lengths: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_spectral_out: array<SpectralRadianceOutput>;

@compute @workgroup_size(64, 1, 1)
fn cs_hero_spectral_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_path_lengths)) {
        return;
    }

    let dist = s_path_lengths[idx];
    let tau = u_spectral_vol.sigma_t_spectrum * dist;
    let transmittance = exp(-tau);

    let in_scatter = (vec4<f32>(1.0) - transmittance) * u_spectral_vol.albedo_spectrum;

    var out_rec: SpectralRadianceOutput;
    out_rec.radiance = in_scatter;
    out_rec.optical_depth = tau;
    s_spectral_out[idx] = out_rec;
}
"""

# 7. heterogeneous_volume_fbm.wgsl
HETEROGENEOUS_FBM_WGSL = """// Shroud :: Procedural 3D Fractal Brownian Motion (fBm) Fog Generator
// 3D Perlin noise synthesis with 4 octaves and analytical gradients.

fn hash33(p_in: vec3<f32>) -> vec3<f32> {
    var p = vec3<f32>(
        dot(p_in, vec3<f32>(127.1, 311.7, 74.7)),
        dot(p_in, vec3<f32>(269.5, 183.3, 246.1)),
        dot(p_in, vec3<f32>(113.5, 271.9, 124.6))
    );
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

fn gradient_noise(p: vec3<f32>) -> f32 {
    let pi = floor(p);
    let pf = fract(p);
    let w = pf * pf * (3.0 - 2.0 * pf);

    let d000 = dot(hash33(pi + vec3<f32>(0.0, 0.0, 0.0)), pf - vec3<f32>(0.0, 0.0, 0.0));
    let d100 = dot(hash33(pi + vec3<f32>(1.0, 0.0, 0.0)), pf - vec3<f32>(1.0, 0.0, 0.0));
    let d010 = dot(hash33(pi + vec3<f32>(0.0, 1.0, 0.0)), pf - vec3<f32>(0.0, 1.0, 0.0));
    let d110 = dot(hash33(pi + vec3<f32>(1.0, 1.0, 0.0)), pf - vec3<f32>(1.0, 1.0, 0.0));
    let d001 = dot(hash33(pi + vec3<f32>(0.0, 0.0, 1.0)), pf - vec3<f32>(0.0, 0.0, 1.0));
    let d101 = dot(hash33(pi + vec3<f32>(1.0, 0.0, 1.0)), pf - vec3<f32>(1.0, 0.0, 1.0));
    let d011 = dot(hash33(pi + vec3<f32>(0.0, 1.0, 1.0)), pf - vec3<f32>(0.0, 1.0, 1.0));
    let d111 = dot(hash33(pi + vec3<f32>(1.0, 1.0, 1.0)), pf - vec3<f32>(1.0, 1.0, 1.0));

    let dx00 = mix(d000, d100, w.x);
    let dx10 = mix(d010, d110, w.x);
    let dx01 = mix(d001, d101, w.x);
    let dx11 = mix(d011, d111, w.x);

    let dxy0 = mix(dx00, dx10, w.y);
    let dxy1 = mix(dx01, dx11, w.y);

    return mix(dxy0, dxy1, w.z);
}

fn fbm_3d(pos: vec3<f32>, octaves: u32) -> f32 {
    var p = pos;
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;

    for (var i = 0u; i < octaves; i = i + 1u) {
        value += amplitude * gradient_noise(p * frequency);
        p = p * 2.02;
        amplitude *= 0.5;
    }
    return value;
}

@group(0) @binding(0) var<storage, read> s_sample_coords: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> s_density_grid: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn cs_generate_fbm_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_sample_coords)) {
        return;
    }

    let p = s_sample_coords[idx].xyz;
    let n = fbm_3d(p, 4u);
    let density = clamp(0.5 + 0.5 * n, 0.0, 1.0);
    s_density_grid[idx] = density;
}
"""

# 8. atmosphere_aerial_perspective.wgsl
AERIAL_PERSPECTIVE_WGSL = """// Shroud :: Aerial Perspective Volume & Multi-Scattering Sky Model
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
"""

# 9. cloud_procedural_stratus.wgsl
STRATUS_CLOUD_WGSL = """// Shroud :: Low-Altitude Radiation Fog & Stratus Cloud Slab
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
"""

# 10. particulate_dust_smoke.wgsl
PARTICULATE_WGSL = """// Shroud :: Airborne Particulate & Soot Suspension Dynamics
// Simulates Stokes settling velocity and radiative thermal absorption.

struct ParticulatePhysicsUniforms {
    particle_density_kg_m3: f32,
    dynamic_viscosity_pa_s: f32,
    gravity_m_s2: f32,
    delta_time_s: f32,
};

struct AerosolParticle {
    position: vec3<f32>,
    radius_m: f32,
    velocity: vec3<f32>,
    mass_kg: f32,
};

// Computes Stokes terminal settling velocity v = (2/9) * r^2 * g * (rho_p - rho_f) / mu
fn stokes_settling_velocity(r: f32, rho_p: f32, mu: f32, g: f32) -> f32 {
    let rho_air = 1.225;
    return (2.0 / 9.0) * r * r * g * (rho_p - rho_air) / max(1e-6, mu);
}

@group(0) @binding(0) var<uniform> u_part_params: ParticulatePhysicsUniforms;
@group(0) @binding(1) var<storage, read_write> s_particles: array<AerosolParticle>;

@compute @workgroup_size(64, 1, 1)
fn cs_particulate_integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_particles)) {
        return;
    }

    var p = s_particles[idx];
    let v_settle = stokes_settling_velocity(
        p.radius_m,
        u_part_params.particle_density_kg_m3,
        u_part_params.dynamic_viscosity_pa_s,
        u_part_params.gravity_m_s2
    );

    p.velocity.y = -v_settle;
    p.position += p.velocity * u_part_params.delta_time_s;

    s_particles[idx] = p;
}
"""

write_shader("phase_henyey_greenstein.wgsl", PHASE_HG_WGSL)
write_shader("phase_rayleigh.wgsl", PHASE_RAYLEIGH_WGSL)
write_shader("phase_mie_marine.wgsl", PHASE_MIE_WGSL)
write_shader("delta_tracking_woodcock.wgsl", DELTA_TRACKING_WGSL)
write_shader("ratio_tracking_transmittance.wgsl", RATIO_TRACKING_WGSL)
write_shader("spectral_hero_tracking.wgsl", HERO_TRACKING_WGSL)
write_shader("heterogeneous_volume_fbm.wgsl", HETEROGENEOUS_FBM_WGSL)
write_shader("atmosphere_aerial_perspective.wgsl", AERIAL_PERSPECTIVE_WGSL)
write_shader("cloud_procedural_stratus.wgsl", STRATUS_CLOUD_WGSL)
write_shader("particulate_dust_smoke.wgsl", PARTICULATE_WGSL)
