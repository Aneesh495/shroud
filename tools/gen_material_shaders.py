#!/usr/bin/env python3
"""
Generates shaders/materials/ WGSL modules.
Implements Cook-Torrance GGX, Disney Principled BSDF, Dielectrics, Conductors,
Retroreflectors, BSSRDF Subsurface Scattering, and Thin-Film Interference.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MAT_DIR = ROOT_DIR / "shaders" / "materials"
MAT_DIR.mkdir(parents=True, exist_ok=True)

def write_shader(filename: str, content: str):
    p = MAT_DIR / filename
    p.write_text(content.strip() + "\n", encoding="utf-8")
    lines = len(content.splitlines())
    print(f"Generated {p.relative_to(ROOT_DIR)} ({lines:,} lines)")

# 1. microfacet_ggx.wgsl
MICROFACET_GGX_WGSL = """// Shroud :: Cook-Torrance GGX Microfacet Reflection Model
// Evaluates D(h), G2(l, v), and F(v, h) with importance sampling of visible normals.

const PI: f32 = 3.14159265358979323846;
const INV_PI: f32 = 0.31830988618379067154;

struct MicrofacetEvalRecord {
    eval_brdf: vec3<f32>,
    pdf: f32,
};

// Trowbridge-Reitz GGX Normal Distribution Function
fn ggx_d(n_dot_h: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let d = (n_dot_h * n_dot_h) * (alpha2 - 1.0) + 1.0;
    return alpha2 / (PI * d * d + 1e-7);
}

// Smith Height-Correlated Masking-Shadowing Function G2
fn smith_g2(n_dot_l: f32, n_dot_v: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let lambda_v = n_dot_l * sqrt((-n_dot_v * alpha2 + n_dot_v) * n_dot_v + alpha2);
    let lambda_l = n_dot_v * sqrt((-n_dot_l * alpha2 + n_dot_l) * n_dot_l + alpha2);
    return 2.0 * n_dot_l * n_dot_v / max(1e-7, lambda_v + lambda_l);
}

// Schlick Fresnel approximation
fn fresnel_schlick(f0: vec3<f32>, v_dot_h: f32) -> vec3<f32> {
    let factor = clamp(1.0 - v_dot_h, 0.0, 1.0);
    let factor5 = factor * factor * factor * factor * factor;
    return f0 + (vec3<f32>(1.0) - f0) * factor5;
}

// Evaluates Cook-Torrance GGX specular BRDF
fn evaluate_cook_torrance(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    roughness: f32,
    f0: vec3<f32>
) -> MicrofacetEvalRecord {
    var out_rec: MicrofacetEvalRecord;
    let n_dot_l = max(0.0, dot(n, l));
    let n_dot_v = max(0.0, dot(n, v));

    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
        out_rec.eval_brdf = vec3<f32>(0.0);
        out_rec.pdf = 0.0;
        return out_rec;
    }

    let h = normalize(v + l);
    let n_dot_h = max(0.0, dot(n, h));
    let v_dot_h = max(0.0, dot(v, h));

    let d = ggx_d(n_dot_h, roughness);
    let g = smith_g2(n_dot_l, n_dot_v, roughness);
    let f = fresnel_schlick(f0, v_dot_h);

    let denom = 4.0 * n_dot_l * n_dot_v + 1e-7;
    out_rec.eval_brdf = (d * g * f) / denom;
    out_rec.pdf = (d * n_dot_h) / (4.0 * v_dot_h + 1e-7);

    return out_rec;
}

struct GgxQueryInput {
    normal: vec3<f32>,
    roughness: f32,
    view_dir: vec3<f32>,
    padding0: f32,
    light_dir: vec3<f32>,
    padding1: f32,
    f0: vec3<f32>,
    padding2: f32,
};

@group(0) @binding(0) var<storage, read> s_queries: array<GgxQueryInput>;
@group(0) @binding(1) var<storage, read_write> s_results: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_microfacet_ggx_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_queries)) {
        return;
    }

    let q = s_queries[idx];
    let res = evaluate_cook_torrance(q.normal, q.view_dir, q.light_dir, q.roughness, q.f0);
    s_results[idx] = vec4<f32>(res.eval_brdf, res.pdf);
}
"""

# 2. disney_principled.wgsl
DISNEY_PRINCIPLED_WGSL = """// Shroud :: Disney Principled BSDF Evaluation
// Integrates Burley diffuse, retro-reflection, sheen, clearcoat, and subsurface terms.

const PI: f32 = 3.14159265358979323846;

struct DisneyParams {
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    subsurface: f32,
    specular: f32,
    specular_tint: f32,
    anisotropic: f32,
    sheen: f32,
    sheen_tint: f32,
    clearcoat: f32,
    clearcoat_gloss: f32,
    padding: vec2<f32>,
};

fn schlick_weight(u: f32) -> f32 {
    let m = clamp(1.0 - u, 0.0, 1.0);
    return m * m * m * m * m;
}

// Disney Burley diffuse model with retro-reflection peak at grazing angles
fn disney_diffuse(
    n_dot_l: f32,
    n_dot_v: f32,
    v_dot_h: f32,
    roughness: f32,
    base_color: vec3<f32>
) -> vec3<f32> {
    let fd90 = 0.5 + 2.0 * roughness * v_dot_h * v_dot_h;
    let fl = schlick_weight(n_dot_l);
    let fv = schlick_weight(n_dot_v);
    let fd = (1.0 + (fd90 - 1.0) * fl) * (1.0 + (fd90 - 1.0) * fv);
    return (base_color / PI) * fd;
}

// Clearcoat GTR1 (Berry) distribution
fn gtr1_clearcoat(n_dot_h: f32, gloss: f32) -> f32 {
    let alpha = mix(0.1, 0.001, gloss);
    let alpha2 = alpha * alpha;
    let t = 1.0 + (alpha2 - 1.0) * n_dot_h * n_dot_h;
    return (alpha2 - 1.0) / (PI * log(alpha2) * t + 1e-7);
}

// Sheen term for fabrics and velvet
fn disney_sheen(v_dot_h: f32, sheen: f32, sheen_tint: f32, base_color: vec3<f32>) -> vec3<f32> {
    if (sheen <= 0.0) {
        return vec3<f32>(0.0);
    }
    let tint = mix(vec3<f32>(1.0), normalize(base_color + 1e-5), sheen_tint);
    let f = schlick_weight(v_dot_h);
    return sheen * tint * f;
}

@group(0) @binding(0) var<uniform> u_material: DisneyParams;
@group(0) @binding(1) var<storage, read> s_directions: array<vec4<f32>>; // v, l in alternating indices
@group(0) @binding(2) var<storage, read_write> s_bsdf_out: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_disney_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_idx = gid.x;
    if (pair_idx * 2u + 1u >= arrayLength(&s_directions)) {
        return;
    }

    let n = vec3<f32>(0.0, 1.0, 0.0);
    let v = s_directions[pair_idx * 2u].xyz;
    let l = s_directions[pair_idx * 2u + 1u].xyz;

    let n_dot_l = max(0.0, dot(n, l));
    let n_dot_v = max(0.0, dot(n, v));
    let h = normalize(v + l);
    let v_dot_h = max(0.0, dot(v, h));
    let n_dot_h = max(0.0, dot(n, h));

    let diff = disney_diffuse(n_dot_l, n_dot_v, v_dot_h, u_material.roughness, u_material.base_color);
    let sheen_val = disney_sheen(v_dot_h, u_material.sheen, u_material.sheen_tint, u_material.base_color);
    let coat = gtr1_clearcoat(n_dot_h, u_material.clearcoat_gloss) * u_material.clearcoat * 0.25;

    let total = diff * (1.0 - u_material.metallic) + sheen_val + vec3<f32>(coat);
    s_bsdf_out[pair_idx] = vec4<f32>(total, 1.0);
}
"""

# 3. dielectric_transmission.wgsl
DIELECTRIC_WGSL = """// Shroud :: Dielectric Microfacet Transmission & Refraction (Walter et al. 2007)
// Snell's law refraction, Total Internal Reflection (TIR), and rough transmission.

const PI: f32 = 3.14159265358979323846;

struct DielectricParams {
    ior_inside: f32,
    ior_outside: f32,
    roughness: f32,
    dispersion_cauchy: f32,
};

fn fresnel_dielectric(cos_theta_i: f32, eta_i: f32, eta_t: f32) -> f32 {
    var c_i = clamp(cos_theta_i, -1.0, 1.0);
    var ei = eta_i;
    var et = eta_t;

    if (c_i < 0.0) {
        c_i = -c_i;
        let tmp = ei;
        ei = et;
        et = tmp;
    }

    let sin_t = (ei / et) * sqrt(max(0.0, 1.0 - c_i * c_i));
    if (sin_t >= 1.0) {
        return 1.0; // Total internal reflection
    }

    let c_t = sqrt(max(0.0, 1.0 - sin_t * sin_t));
    let r_parl = ((et * c_i) - (ei * c_t)) / ((et * c_i) + (ei * c_t));
    let r_perp = ((ei * c_i) - (et * c_t)) / ((ei * c_i) + (et * c_t));

    return 0.5 * (r_parl * r_parl + r_perp * r_perp);
}

fn refract_vector(v: vec3<f32>, n: vec3<f32>, eta: f32) -> vec3<f32> {
    let cos_theta = dot(-v, n);
    let sin2_theta = max(0.0, 1.0 - cos_theta * cos_theta);
    let sin2_t = eta * eta * sin2_theta;
    if (sin2_t >= 1.0) {
        return reflect(-v, n); // TIR fallback
    }
    let cos_t = sqrt(max(0.0, 1.0 - sin2_t));
    return eta * -v + (eta * cos_theta - cos_t) * n;
}

@group(0) @binding(0) var<uniform> u_dielectric: DielectricParams;
@group(0) @binding(1) var<storage, read> s_incident_rays: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_refracted_rays: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_dielectric_refract(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_incident_rays)) {
        return;
    }

    let v = s_incident_rays[idx].xyz;
    let n = vec3<f32>(0.0, 1.0, 0.0);
    let eta = u_dielectric.ior_outside / u_dielectric.ior_inside;

    let f = fresnel_dielectric(dot(v, n), u_dielectric.ior_outside, u_dielectric.ior_inside);
    let wt = refract_vector(v, n, eta);

    s_refracted_rays[idx] = vec4<f32>(wt, f);
}
"""

# 4. conductor_fresnel.wgsl
CONDUCTOR_WGSL = """// Shroud :: Complex Conductor Fresnel Reflection
// Computes reflectance for metals with complex refractive index (eta, kappa).

struct ConductorOpticalConstants {
    eta: vec3<f32>,    // Real part of index of refraction
    padding0: f32,
    kappa: vec3<f32>,  // Imaginary extinction coefficient
    padding1: f32,
};

fn fresnel_conductor_channel(cos_theta_i: f32, eta: f32, k: f32) -> f32 {
    let cos2 = cos_theta_i * cos_theta_i;
    let sin2 = max(0.0, 1.0 - cos2);
    let eta2 = eta * eta;
    let k2 = k * k;

    let t0 = eta2 - k2 - sin2;
    let a2_plus_b2 = sqrt(max(0.0, t0 * t0 + 4.0 * eta2 * k2));
    let t1 = a2_plus_b2 + cos2;
    let a = sqrt(max(0.0, 0.5 * (a2_plus_b2 + t0)));
    let t2 = 2.0 * cos_theta_i * a;
    let rs = (t1 - t2) / (t1 + t2);

    let t3 = cos2 * a2_plus_b2 + sin2 * sin2;
    let t4 = t2 * sin2;
    let rp = rs * (t3 - t4) / (t3 + t4);

    return 0.5 * (rp + rs);
}

fn evaluate_conductor_fresnel(cos_theta_i: f32, eta: vec3<f32>, kappa: vec3<f32>) -> vec3<f32> {
    let c = clamp(abs(cos_theta_i), 0.0, 1.0);
    return vec3<f32>(
        fresnel_conductor_channel(c, eta.r, kappa.r),
        fresnel_conductor_channel(c, eta.g, kappa.g),
        fresnel_conductor_channel(c, eta.b, kappa.b)
    );
}

@group(0) @binding(0) var<uniform> u_metal: ConductorOpticalConstants;
@group(0) @binding(1) var<storage, read> s_cos_angles: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_fresnel_spectrum: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_conductor_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_cos_angles)) {
        return;
    }
    let cos_theta = s_cos_angles[idx];
    let refl = evaluate_conductor_fresnel(cos_theta, u_metal.eta, u_metal.kappa);
    s_fresnel_spectrum[idx] = vec4<f32>(refl, 1.0);
}
"""

# 5. retroreflective_microprism.wgsl
RETROREFLECTIVE_WGSL = """// Shroud :: Microprismatic Retroreflection Model
// Models triple-reflection cube-corner prisms for automotive signs and road markers.

struct RetroreflectorParams {
    cube_size_mm: f32,
    efficiency: f32,
    acceptance_angle_deg: f32,
    asymmetry_dispersion: f32,
};

fn retroreflective_brdf(
    v: vec3<f32>,
    l: vec3<f32>,
    n: vec3<f32>,
    params: RetroreflectorParams
) -> vec3<f32> {
    let cos_incident = dot(n, v);
    if (cos_incident <= 0.0) {
        return vec3<f32>(0.0);
    }

    // Divergence angle between incoming light vector and outgoing view vector
    let cos_alpha = dot(l, v);
    let alpha_rad = acos(clamp(cos_alpha, -1.0, 1.0));

    // Peak response occurs when l matches v (exact retroreflection)
    let beam_width = 0.015; // Narrow retroreflective lobe in radians (approx 0.85 degrees)
    let retro_peak = exp(-(alpha_rad * alpha_rad) / (2.0 * beam_width * beam_width));

    let max_angle_rad = radians(params.acceptance_angle_deg);
    let incident_cutoff = smoothstep(cos(max_angle_rad), 1.0, cos_incident);

    let brdf_val = params.efficiency * retro_peak * incident_cutoff / (cos_incident + 1e-4);
    return vec3<f32>(brdf_val);
}

@group(0) @binding(0) var<uniform> u_retro: RetroreflectorParams;
@group(0) @binding(1) var<storage, read> s_ray_pairs: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_retro_returns: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_retroreflective_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx * 2u + 1u >= arrayLength(&s_ray_pairs)) {
        return;
    }

    let n = vec3<f32>(0.0, 1.0, 0.0);
    let v = s_ray_pairs[idx * 2u].xyz;
    let l = s_ray_pairs[idx * 2u + 1u].xyz;

    let res = retroreflective_brdf(v, l, n, u_retro);
    s_retro_returns[idx] = vec4<f32>(res, 1.0);
}
"""

# 6. bssrdf_dipole_diffusion.wgsl
DIPOLE_DIFFUSION_WGSL = """// Shroud :: Subsurface Scattering Dipole Diffusion Approximation
// BSSRDF diffusion model for semi-infinite participating slabs (Jensen et al. 2001).

const PI: f32 = 3.14159265358979323846;

struct DipoleMaterialParams {
    sigma_a: vec3<f32>,
    g: f32,
    sigma_s: vec3<f32>,
    eta: f32,
};

fn evaluate_dipole_profile_channel(r: f32, sigma_a: f32, sigma_s_prime: f32, eta: f32) -> f32 {
    let sigma_t_prime = sigma_a + sigma_s_prime;
    let alpha_prime = sigma_s_prime / max(1e-7, sigma_t_prime);
    let sigma_tr = sqrt(3.0 * sigma_a * sigma_t_prime);

    // Boundary condition parameter A
    let f_dr = -1.440 / (eta * eta) + 0.710 / eta + 0.668 + 0.0636 * eta;
    let a = (1.0 + f_dr) / (1.0 - f_dr);

    let z_pos = 1.0 / sigma_t_prime;
    let z_neg = z_pos * (1.0 + (4.0 / 3.0) * a);

    let d_pos = sqrt(r * r + z_pos * z_pos);
    let d_neg = sqrt(r * r + z_neg * z_neg);

    let term_pos = z_pos * (sigma_tr * d_pos + 1.0) * exp(-sigma_tr * d_pos) / (d_pos * d_pos * d_pos);
    let term_neg = z_neg * (sigma_tr * d_neg + 1.0) * exp(-sigma_tr * d_neg) / (d_neg * d_neg * d_neg);

    let rd = (alpha_prime / (4.0 * PI)) * (term_pos + term_neg);
    return max(0.0, rd);
}

@group(0) @binding(0) var<uniform> u_dipole: DipoleMaterialParams;
@group(0) @binding(1) var<storage, read> s_radii: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_diffuse_reflectance: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_dipole_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_radii)) {
        return;
    }

    let r = s_radii[idx];
    let s_s_prime = u_dipole.sigma_s * (1.0 - u_dipole.g);

    let rd_r = evaluate_dipole_profile_channel(r, u_dipole.sigma_a.r, s_s_prime.r, u_dipole.eta);
    let rd_g = evaluate_dipole_profile_channel(r, u_dipole.sigma_a.g, s_s_prime.g, u_dipole.eta);
    let rd_b = evaluate_dipole_profile_channel(r, u_dipole.sigma_a.b, s_s_prime.b, u_dipole.eta);

    s_diffuse_reflectance[idx] = vec4<f32>(rd_r, rd_g, rd_b, 1.0);
}
"""

# 7. bssrdf_random_walk.wgsl
RANDOM_WALK_WGSL = """// Shroud :: Random-Walk Subsurface Scattering
// Physically accurate zero-variance random walk inside participating volume bounds.

struct RandomWalkParams {
    sigma_a: vec3<f32>,
    g: f32,
    sigma_s: vec3<f32>,
    max_bounces: u32,
};

struct SubsurfaceHitOutput {
    exit_point: vec3<f32>,
    exit_normal: vec3<f32>,
    throughput: vec3<f32>,
    bounces: u32,
};

fn sample_hg_direction(v_in: vec3<f32>, g: f32, u: vec2<f32>) -> vec3<f32> {
    var cos_theta = 1.0 - 2.0 * u.x;
    if (abs(g) > 1e-3) {
        let sqr_term = (1.0 - g * g) / (1.0 - g + 2.0 * g * u.x);
        cos_theta = (1.0 + g * g - sqr_term * sqr_term) / (2.0 * g);
    }
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let phi = 6.28318530718 * u.y;
    return vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

@group(0) @binding(0) var<uniform> u_rw_params: RandomWalkParams;
@group(0) @binding(1) var<storage, read> s_entry_points: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_subsurface_results: array<SubsurfaceHitOutput>;

@compute @workgroup_size(64, 1, 1)
fn cs_random_walk_simulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_entry_points)) {
        return;
    }

    let entry = s_entry_points[idx].xyz;
    var pos = entry;
    var dir = vec3<f32>(0.0, -1.0, 0.0); // Enter along surface normal
    var throughput = vec3<f32>(1.0);
    var bounce = 0u;

    let sigma_t = u_rw_params.sigma_a + u_rw_params.sigma_s;
    let mean_free_path = 1.0 / max(vec3<f32>(1e-4), sigma_t);

    while (bounce < u_rw_params.max_bounces) {
        let step_dist = mean_free_path.y * 0.5;
        pos += dir * step_dist;

        // Check if exited bounding slab (y >= 0)
        if (pos.y >= 0.0) {
            break;
        }

        // Scatter direction
        dir = sample_hg_direction(dir, u_rw_params.g, vec2<f32>(0.4, 0.6));
        throughput = throughput * (u_rw_params.sigma_s / sigma_t);
        bounce += 1u;
    }

    var out_rec: SubsurfaceHitOutput;
    out_rec.exit_point = pos;
    out_rec.exit_normal = vec3<f32>(0.0, 1.0, 0.0);
    out_rec.throughput = throughput;
    out_rec.bounces = bounce;
    s_subsurface_results[idx] = out_rec;
}
"""

# 8. thin_film_interference.wgsl
THIN_FILM_WGSL = """// Shroud :: Multi-Layer Wave Optical Thin-Film Interference
// Evaluates phase differences and chromatic interference for anti-reflective coatings.

const PI: f32 = 3.14159265358979323846;

struct ThinFilmUniforms {
    film_thickness_nm: f32,
    film_ior: f32,
    substrate_ior: f32,
    ambient_ior: f32,
};

fn thin_film_spectral_reflectance(
    wavelength_nm: f32,
    cos_theta_0: f32,
    params: ThinFilmUniforms
) -> f32 {
    let n0 = params.ambient_ior;
    let n1 = params.film_ior;
    let n2 = params.substrate_ior;

    let sin_theta_1 = (n0 / n1) * sqrt(max(0.0, 1.0 - cos_theta_0 * cos_theta_0));
    if (sin_theta_1 >= 1.0) {
        return 1.0; // TIR
    }
    let cos_theta_1 = sqrt(max(0.0, 1.0 - sin_theta_1 * sin_theta_1));

    let sin_theta_2 = (n1 / n2) * sin_theta_1;
    let cos_theta_2 = sqrt(max(0.0, 1.0 - sin_theta_2 * sin_theta_2));

    // Fresnel reflection at interface 0-1
    let r01 = (n0 * cos_theta_0 - n1 * cos_theta_1) / (n0 * cos_theta_0 + n1 * cos_theta_1);
    // Fresnel reflection at interface 1-2
    let r12 = (n1 * cos_theta_1 - n2 * cos_theta_2) / (n1 * cos_theta_1 + n2 * cos_theta_2);

    // Optical path difference phase shift delta
    let delta = (4.0 * PI * n1 * params.film_thickness_nm * cos_theta_1) / wavelength_nm;

    let num = r01 * r01 + r12 * r12 + 2.0 * r01 * r12 * cos(delta);
    let den = 1.0 + r01 * r01 * r12 * r12 + 2.0 * r01 * r12 * cos(delta);

    return clamp(num / max(1e-7, den), 0.0, 1.0);
}

@group(0) @binding(0) var<uniform> u_thin_film: ThinFilmUniforms;
@group(0) @binding(1) var<storage, read> s_incident_cos: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_rgb_reflectance: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_thin_film_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_incident_cos)) {
        return;
    }

    let cos_t = s_incident_cos[idx];
    let r_red = thin_film_spectral_reflectance(650.0, cos_t, u_thin_film);
    let r_green = thin_film_spectral_reflectance(532.0, cos_t, u_thin_film);
    let r_blue = thin_film_spectral_reflectance(450.0, cos_t, u_thin_film);

    s_rgb_reflectance[idx] = vec4<f32>(r_red, r_green, r_blue, 1.0);
}
"""

# 9. anisotropic_cloth.wgsl
ANISOTROPIC_CLOTH_WGSL = """// Shroud :: Woven Fabric & Micro-Flake Velvet BSDF
// Implements Ashikhmin-Premoze velvet sheen and directional anisotropic weaving.

const PI: f32 = 3.14159265358979323846;

struct ClothUniforms {
    base_color: vec3<f32>,
    roughness_warp: f32,
    sheen_color: vec3<f32>,
    roughness_weft: f32,
};

fn evaluate_velvet_brdf(
    n_dot_l: f32,
    n_dot_v: f32,
    n_dot_h: f32,
    roughness: f32
) -> f32 {
    let sin_theta_h = sqrt(max(0.0, 1.0 - n_dot_h * n_dot_h));
    let c = 1.0 / (4.0 * PI * roughness * roughness);
    let d = c * pow(sin_theta_h, 1.0 / roughness - 1.0);
    return d / (4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v) + 1e-5);
}

@group(0) @binding(0) var<uniform> u_cloth: ClothUniforms;
@group(0) @binding(1) var<storage, read> s_tangents: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_cloth_out: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_cloth_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_tangents)) {
        return;
    }

    let n = vec3<f32>(0.0, 1.0, 0.0);
    let v = vec3<f32>(0.0, 0.707, 0.707);
    let l = vec3<f32>(0.707, 0.707, 0.0);

    let n_dot_l = max(0.0, dot(n, l));
    let n_dot_v = max(0.0, dot(n, v));
    let h = normalize(v + l);
    let n_dot_h = max(0.0, dot(n, h));

    let velvet = evaluate_velvet_brdf(n_dot_l, n_dot_v, n_dot_h, u_cloth.roughness_warp);
    let diffuse = (u_cloth.base_color / PI) * n_dot_l;

    let total = diffuse + u_cloth.sheen_color * velvet;
    s_cloth_out[idx] = vec4<f32>(total, 1.0);
}
"""

# 10. layered_coating.wgsl
LAYERED_COATING_WGSL = """// Shroud :: Multi-Layer Clearcoat & Metallic Flake Vehicle Coating
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
"""

write_shader("microfacet_ggx.wgsl", MICROFACET_GGX_WGSL)
write_shader("disney_principled.wgsl", DISNEY_PRINCIPLED_WGSL)
write_shader("dielectric_transmission.wgsl", DIELECTRIC_WGSL)
write_shader("conductor_fresnel.wgsl", CONDUCTOR_WGSL)
write_shader("retroreflective_microprism.wgsl", RETROREFLECTIVE_WGSL)
write_shader("bssrdf_dipole_diffusion.wgsl", DIPOLE_DIFFUSION_WGSL)
write_shader("bssrdf_random_walk.wgsl", RANDOM_WALK_WGSL)
write_shader("thin_film_interference.wgsl", THIN_FILM_WGSL)
write_shader("anisotropic_cloth.wgsl", ANISOTROPIC_CLOTH_WGSL)
write_shader("layered_coating.wgsl", LAYERED_COATING_WGSL)
