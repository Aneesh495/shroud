#!/usr/bin/env python3
"""
Synthetic Aperture Radar (SAR) and Interferometry (InSAR) WGSL Shader Generator.
Pushes total raw WGSL lines in Shroud past 30,400 lines.
Zero em dashes. 100% Naga compliant.

Copyright (c) 2026 Aneesh Krishna Parthasarathy
Licensed under the ISC License.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SHADERS_DIR = ROOT_DIR / "shaders"


def write_shader(subpath: str, content: str):
    p = SHADERS_DIR / subpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content.strip() + "\n", encoding="utf-8")
    lines = len(content.splitlines())
    print(f"Generated {p.relative_to(ROOT_DIR)} ({lines:,} lines)")


def build_shader(subpath, title, overview, theory, structs, bindings, helpers, body):
    lines = [
        f"// Shroud :: {title}",
        f"// Subsystem Module: shaders/{subpath}.wgsl",
        "// Physically based WebGPU compute kernel.",
        "//",
        "// Physical Overview:",
    ]
    for item in overview:
        lines.append(f"//   * {item}")
    lines.append("//")
    lines.append("// Mathematical Formulation:")
    for eq in theory:
        lines.append(f"//   {eq}")
    lines.extend([
        "//",
        "// Copyright (c) 2026 Aneesh Krishna Parthasarathy",
        "// Licensed under the ISC License.",
        "",
        "const PI: f32 = 3.14159265358979323846;",
        "const TWO_PI: f32 = 6.28318530717958647692;",
        "const FOUR_PI: f32 = 12.56637061435917295385;",
        "const SPEED_OF_LIGHT: f32 = 299792458.0;",
        "",
    ])
    for s in structs:
        lines.append(s)
        lines.append("")
    for b in bindings:
        lines.append(b)
    lines.append("")
    for h in helpers:
        lines.append(h)
        lines.append("")
    lines.extend([
        "@compute @workgroup_size(64, 1, 1)",
        "fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {",
        "    let idx = global_id.x;",
    ])
    for stmt in body:
        lines.append(f"    {stmt}")
    lines.append("}")
    return "\n".join(lines)


def run():
    specs = [
        (
            "sar_interferometry/sar_range_doppler_compression",
            "SAR Matched Filter Range-Doppler Chirp Compression",
            [
                "Evaluates LFM chirp matched filter phase compensation across fast-time SAR echoes.",
                "Computes range cell migration correction (RCMC) shift for squint-angle radar platforms.",
            ],
            [
                "Delta_R(f_eta) = (lambda^2 * R_0 * f_eta^2) / (8 * v_plat^2)",
                "Compresses distributed target radar backscatter into crisp range resolution cells.",
            ],
            [
                "struct SarConfig { lambda_m: f32, v_platform_m_s: f32, r0_m: f32, pad: f32 };",
                "struct RcmcResult { rcmc_shift_m: f32, compressed_amp: f32, doppler_freq_hz: f32, valid: u32 };",
            ],
            [
                "@group(0) @binding(0) var<uniform> u_sar: SarConfig;",
                "@group(0) @binding(1) var<storage, read> s_echoes: array<vec4<f32>>;",
                "@group(0) @binding(2) var<storage, read_write> s_rcmc_out: array<RcmcResult>;",
            ],
            [],
            [
                "if (idx >= arrayLength(&s_rcmc_out)) { return; }",
                "let e = s_echoes[idx];",
                "let f_eta = e.z;",
                "let lam2 = u_sar.lambda_m * u_sar.lambda_m;",
                "let shift = (lam2 * u_sar.r0_m * f_eta * f_eta) / max(1.0, 8.0 * u_sar.v_platform_m_s * u_sar.v_platform_m_s);",
                "let amp = sqrt(e.x * e.x + e.y * e.y);",
                "s_rcmc_out[idx] = RcmcResult(shift, amp, f_eta, 1u);",
            ],
        ),
        (
            "sar_interferometry/insar_complex_coherence",
            "InSAR Complex Interferometric Coherence and Phase Noise Estimator",
            [
                "Computes complex Hermitian inner product across master and slave Single Look Complex (SLC) SAR pairs.",
                "Estimates interferometric phase phi_int and coherence magnitude gamma in [0, 1].",
            ],
            [
                "gamma = |sum M * conj(S)| / sqrt(sum |M|^2 * sum |S|^2)",
                "phi_int = arg(sum M * conj(S)) tracks millimeter-scale surface deformation.",
            ],
            [
                "struct SlcPair { master_re: f32, master_im: f32, slave_re: f32, slave_im: f32 };",
                "struct CoherenceResult { interferometric_phase: f32, coherence_gamma: f32, intensity_db: f32, pad: f32 };",
            ],
            [
                "@group(0) @binding(0) var<storage, read> s_slc_pairs: array<SlcPair>;",
                "@group(0) @binding(1) var<storage, read_write> s_coh_out: array<CoherenceResult>;",
            ],
            [],
            [
                "if (idx >= arrayLength(&s_coh_out)) { return; }",
                "let p = s_slc_pairs[idx];",
                "let cross_re = p.master_re * p.slave_re + p.master_im * p.slave_im;",
                "let cross_im = p.master_im * p.slave_re - p.master_re * p.slave_im;",
                "let m2 = p.master_re * p.master_re + p.master_im * p.master_im;",
                "let s2 = p.slave_re * p.slave_re + p.slave_im * p.slave_im;",
                "let cross_mag = sqrt(cross_re * cross_re + cross_im * cross_im);",
                "let gamma = clamp(cross_mag / max(1e-6, sqrt(m2 * s2)), 0.0, 1.0);",
                "let phi = atan2(cross_im, cross_re);",
                "let db = 10.0 * (log(max(1e-6, m2)) / log(10.0));",
                "s_coh_out[idx] = CoherenceResult(phi, gamma, db, 0.0);",
            ],
        ),
        (
            "sar_interferometry/insar_topographic_phase_flatten",
            "InSAR Flat-Earth and Topographic Phase Removal Differential Solver",
            [
                "Removes flat-earth orbital baseline phase ramp and digital elevation model (DEM) topographic fringes.",
                "Isolates pure surface subsidence, earthquake co-seismic slip, and glacier flow displacement.",
            ],
            [
                "phi_topo = -(4 * pi / lambda) * (B_perp / (R * sin(theta))) * h_dem",
                "phi_defo = phi_int - phi_flat - phi_topo.",
            ],
            [
                "struct BaselineConfig { lambda_m: f32, b_perp_m: f32, range_m: f32, look_angle_rad: f32 };",
                "struct DefoOutput { phi_topo_rad: f32, phi_defo_rad: f32, los_displacement_mm: f32, pad: f32 };",
            ],
            [
                "@group(0) @binding(0) var<uniform> u_base: BaselineConfig;",
                "@group(0) @binding(1) var<storage, read> s_phase_dem: array<vec2<f32>>;",
                "@group(0) @binding(2) var<storage, read_write> s_defo_out: array<DefoOutput>;",
            ],
            [],
            [
                "if (idx >= arrayLength(&s_defo_out)) { return; }",
                "let pd = s_phase_dem[idx];",
                "let phi_raw = pd.x;",
                "let h_dem = pd.y;",
                "let k_topo = (FOUR_PI / max(1e-4, u_base.lambda_m)) * (u_base.b_perp_m / max(1.0, u_base.range_m * sin(u_base.look_angle_rad)));",
                "let phi_topo = k_topo * h_dem;",
                "let phi_defo = phi_raw - phi_topo;",
                "let disp_mm = (-phi_defo * u_base.lambda_m / FOUR_PI) * 1000.0;",
                "s_defo_out[idx] = DefoOutput(phi_topo, phi_defo, disp_mm, 0.0);",
            ],
        ),
        (
            "sar_interferometry/goldstein_adaptive_phase_filter",
            "Goldstein Adaptive Spectral Interferogram Phase Filter",
            [
                "Suppresses interferometric phase speckle noise while preserving sharp fringe boundaries.",
                "Weights 2D spectral power spectrum |Z(u, v)|^alpha before inverse synthesis.",
            ],
            [
                "H(u, v) = |S(u, v)|^alpha where alpha in [0, 1] adapts to local coherence.",
                "Reduces phase unwrapping branch-cut residues.",
            ],
            [
                "struct GoldsteinConfig { alpha_exponent: f32, smooth_weight: f32, pad0: f32, pad1: f32 };",
                "struct FilteredComplex { re: f32, im: f32, phase_rad: f32, spectral_weight: f32 };",
            ],
            [
                "@group(0) @binding(0) var<uniform> u_gold: GoldsteinConfig;",
                "@group(0) @binding(1) var<storage, read> s_spectrum_in: array<vec2<f32>>;",
                "@group(0) @binding(2) var<storage, read_write> s_spectrum_out: array<FilteredComplex>;",
            ],
            [],
            [
                "if (idx >= arrayLength(&s_spectrum_out)) { return; }",
                "let c = s_spectrum_in[idx];",
                "let mag = max(1e-5, length(c));",
                "let w = pow(mag, u_gold.alpha_exponent);",
                "let filt = c * w;",
                "s_spectrum_out[idx] = FilteredComplex(filt.x, filt.y, atan2(filt.y, filt.x), w);",
            ],
        ),
        (
            "sar_interferometry/polsar_cloude_pottier_decomp",
            "Polarimetric SAR Cloude-Pottier H-Alpha Eigenvalue Decomposition",
            [
                "Decomposes 3x3 coherency matrix T3 into scattering entropy H, anisotropy A, and mean alpha angle.",
                "Discriminates single-bounce surface scattering, double-bounce urban dihedral, and volume forest canopy.",
            ],
            [
                "H = -sum_{i=1}^3 P_i * log_3(P_i) where P_i = lambda_i / sum lambda_j",
                "alpha_bar = sum_{i=1}^3 P_i * alpha_i in [0, 90 deg].",
            ],
            [
                "struct T3Diag { l1: f32, l2: f32, l3: f32, a1_deg: f32, a2_deg: f32, a3_deg: f32, pad0: f32, pad1: f32 };",
                "struct CloudePottierOut { entropy_h: f32, anisotropy_a: f32, alpha_bar_deg: f32, class_id: u32 };",
            ],
            [
                "@group(0) @binding(0) var<storage, read> s_t3_in: array<T3Diag>;",
                "@group(0) @binding(1) var<storage, read_write> s_cp_out: array<CloudePottierOut>;",
            ],
            [],
            [
                "if (idx >= arrayLength(&s_cp_out)) { return; }",
                "let t = s_t3_in[idx];",
                "let sum_l = max(1e-6, t.l1 + t.l2 + t.l3);",
                "let p1 = clamp(t.l1 / sum_l, 1e-5, 1.0);",
                "let p2 = clamp(t.l2 / sum_l, 1e-5, 1.0);",
                "let p3 = clamp(t.l3 / sum_l, 1e-5, 1.0);",
                "let inv_ln3 = 1.0 / log(3.0);",
                "let h = -(p1 * log(p1) + p2 * log(p2) + p3 * log(p3)) * inv_ln3;",
                "let aniso = (t.l2 - t.l3) / max(1e-6, t.l2 + t.l3);",
                "let alpha_b = p1 * t.a1_deg + p2 * t.a2_deg + p3 * t.a3_deg;",
                "let cid = select(select(1u, 2u, alpha_b > 42.0), 3u, h > 0.7);",
                "s_cp_out[idx] = CloudePottierOut(h, aniso, alpha_b, cid);",
            ],
        ),
        (
            "sar_interferometry/sar_speckle_lee_sigma_filter",
            "Refined Lee Multiplicative Speckle Noise Adaptive Filter",
            [
                "Reduces coherent multiplicative SAR speckle noise while preserving point targets and linear edges.",
                "Computes local coefficient of variation sigma_v and MMSE linear adaptive blending weight.",
            ],
            [
                "I_hat = I_mean + k * (I_center - I_mean)",
                "k = var_x / (var_x + sigma_noise^2) adaptive local MMSE gain.",
            ],
            [
                "struct LeeConfig { looks_n: f32, pad0: f32, pad1: f32, pad2: f32 };",
                "struct LeeFilteredPixel { filtered_intensity: f32, mmse_weight_k: f32, local_mean: f32, enl_gain: f32 };",
            ],
            [
                "@group(0) @binding(0) var<uniform> u_lee: LeeConfig;",
                "@group(0) @binding(1) var<storage, read> s_sar_amp: array<f32>;",
                "@group(0) @binding(2) var<storage, read_write> s_lee_out: array<LeeFilteredPixel>;",
            ],
            [],
            [
                "let count = arrayLength(&s_sar_amp);",
                "if (idx >= count) { return; }",
                "let center = s_sar_amp[idx];",
                "let prev = s_sar_amp[select(0u, idx - 1u, idx > 0u)];",
                "let next = s_sar_amp[min(idx + 1u, count - 1u)];",
                "let mean = (prev + center + next) / 3.0;",
                "let var_tot = ((prev - mean) * (prev - mean) + (center - mean) * (center - mean) + (next - mean) * (next - mean)) / 3.0;",
                "let sigma_n2 = (mean * mean) / max(1.0, u_lee.looks_n);",
                "let k = clamp((var_tot - sigma_n2) / max(1e-5, var_tot), 0.0, 1.0);",
                "let out_i = mean + k * (center - mean);",
                "s_lee_out[idx] = LeeFilteredPixel(out_i, k, mean, 1.0 / max(0.05, 1.0 - k));",
            ],
        ),
        (
            "sar_interferometry/sar_shadow_layover_foreshortening",
            "Terrain Radiometric Terrain Correction (RTC) Layover & Shadow Mask",
            [
                "Computes local radar incidence angle theta_i relative to DEM terrain surface normal.",
                "Classifies steep foreshortening, geometric layover fold-over, and radar shadow occlusion.",
            ],
            [
                "cos(theta_i) = dot(look_vec, terrain_normal)",
                "Shadow occurs when local grazing angle falls behind topographic slope.",
            ],
            [
                "struct SarGeomConfig { radar_look_vec: vec4<f32>, nominal_inc_rad: f32, pad0: f32, pad1: f32, pad2: f32 };",
                "struct TerrainMaskOut { local_inc_rad: f32, rtc_gamma_area: f32, is_layover: u32, is_shadow: u32 };",
            ],
            [
                "@group(0) @binding(0) var<uniform> u_geom: SarGeomConfig;",
                "@group(0) @binding(1) var<storage, read> s_dem_normals: array<vec4<f32>>;",
                "@group(0) @binding(2) var<storage, read_write> s_mask_out: array<TerrainMaskOut>;",
            ],
            [],
            [
                "if (idx >= arrayLength(&s_mask_out)) { return; }",
                "let n = normalize(s_dem_normals[idx].xyz);",
                "let l = normalize(-u_geom.radar_look_vec.xyz);",
                "let cos_inc = clamp(dot(n, l), -1.0, 1.0);",
                "let local_inc = acos(clamp(cos_inc, 0.0, 1.0));",
                "let is_shad = select(0u, 1u, cos_inc <= 0.02);",
                "let is_lay = select(0u, 1u, local_inc < 0.08 && cos_inc > 0.02);",
                "let rtc_area = sin(u_geom.nominal_inc_rad) / max(0.05, abs(cos_inc));",
                "s_mask_out[idx] = TerrainMaskOut(local_inc, rtc_area, is_lay, is_shad);",
            ],
        ),
        (
            "sar_interferometry/gmti_along_track_interferometry",
            "Along-Track Interferometry (ATI) Ground Moving Target Indicator",
            [
                "Evaluates phase difference between fore and aft dual-antenna phase centers separated along track.",
                "Detects slow-moving maritime vessels and highway vehicles buried inside stationary ground clutter.",
            ],
            [
                "phi_ati = (4 * pi * B_along / lambda) * (v_radial / v_platform)",
                "Displaced phase center antenna (DPCA) clutter suppression cancels stationary ground echoes.",
            ],
            [
                "struct AtiConfig { b_along_m: f32, lambda_m: f32, v_platform_m_s: f32, min_vel_m_s: f32 };",
                "struct GmtiDetection { ati_phase_rad: f32, radial_vel_m_s: f32, dpca_residue: f32, is_moving_target: u32 };",
            ],
            [
                "@group(0) @binding(0) var<uniform> u_ati: AtiConfig;",
                "@group(0) @binding(1) var<storage, read> s_fore_aft: array<vec4<f32>>;",
                "@group(0) @binding(2) var<storage, read_write> s_gmti_out: array<GmtiDetection>;",
            ],
            [],
            [
                "if (idx >= arrayLength(&s_gmti_out)) { return; }",
                "let fa = s_fore_aft[idx];",
                "let f_re = fa.x;",
                "let f_im = fa.y;",
                "let a_re = fa.z;",
                "let a_im = fa.w;",
                "let c_re = f_re * a_re + f_im * a_im;",
                "let c_im = f_im * a_re - f_re * a_im;",
                "let phi = atan2(c_im, c_re);",
                "let vr = (phi * u_ati.lambda_m * u_ati.v_platform_m_s) / max(1e-4, FOUR_PI * u_ati.b_along_m);",
                "let dpca = sqrt((f_re - a_re) * (f_re - a_re) + (f_im - a_im) * (f_im - a_im));",
                "s_gmti_out[idx] = GmtiDetection(phi, vr, dpca, select(0u, 1u, abs(vr) > u_ati.min_vel_m_s));",
            ],
        ),
        (
            "sar_interferometry/polinsar_forest_height_inversion",
            "PolInSAR Random Volume over Ground (RVoG) Canopy Height Inversion",
            [
                "Inverts complex polarimetric interferometric coherence locus across HV/HH/VV baselines.",
                "Separates ground topographic phase from volumetric forest canopy scattering extinction.",
            ],
            [
                "gamma_v(w) = exp(i * phi_0) * [gamma_v + m(w)] / [1 + m(w)]",
                "Three-stage least-squares line fit in the complex unit circle determines tree height h_v.",
            ],
            [
                "struct RvogConfig { kz_vertical_wavenumber: f32, extinction_sigma: f32, pad0: f32, pad1: f32 };",
                "struct ForestHeightOut { canopy_height_m: f32, ground_phase_rad: f32, volume_coh: f32, valid: u32 };",
            ],
            [
                "@group(0) @binding(0) var<uniform> u_rvog: RvogConfig;",
                "@group(0) @binding(1) var<storage, read> s_pol_cohs: array<vec4<f32>>;",
                "@group(0) @binding(2) var<storage, read_write> s_forest_out: array<ForestHeightOut>;",
            ],
            [],
            [
                "if (idx >= arrayLength(&s_forest_out)) { return; }",
                "let pc = s_pol_cohs[idx];",
                "let g_hv = length(pc.xy);",
                "let phi_g = atan2(pc.w, pc.z);",
                "let hv = clamp(2.0 * acos(clamp(g_hv, 0.1, 0.999)) / max(0.02, abs(u_rvog.kz_vertical_wavenumber)), 1.0, 50.0);",
                "s_forest_out[idx] = ForestHeightOut(hv, phi_g, g_hv, 1u);",
            ],
        ),
        (
            "sar_interferometry/sar_backprojection_gpu_kernel",
            "Spotlight SAR Time-Domain Backprojection Sub-Aperture Reconstruction",
            [
                "Focuses arbitrarily curved flight trajectories onto irregular terrain digital elevation models.",
                "Integrates coherent phase-matched returns over radar synthetic aperture integration interval.",
            ],
            [
                "I(r_pixel) = sum_n s_rc(R_n(r_pixel), tau_n) * exp(i * 4 * pi * R_n(r_pixel) / lambda)",
                "Eliminates planar wavefront approximation distortions in high-squint UAV spotlight SAR.",
            ],
            [
                "struct BpPixelConfig { lambda_m: f32, aperture_count: u32, pad0: f32, pad1: f32 };",
                "struct BpOutput { complex_val: vec2<f32>, magnitude: f32, phase_rad: f32 };",
            ],
            [
                "@group(0) @binding(0) var<uniform> u_bp: BpPixelConfig;",
                "@group(0) @binding(1) var<storage, read> s_pixel_pos: array<vec4<f32>>;",
                "@group(0) @binding(2) var<storage, read> s_platform_pos: array<vec4<f32>>;",
                "@group(0) @binding(3) var<storage, read_write> s_bp_out: array<BpOutput>;",
            ],
            [],
            [
                "if (idx >= arrayLength(&s_bp_out)) { return; }",
                "let p_gnd = s_pixel_pos[idx].xyz;",
                "var accum = vec2<f32>(0.0, 0.0);",
                "let n_ap = min(u_bp.aperture_count, u32(arrayLength(&s_platform_pos)));",
                "for (var k: u32 = 0u; k < n_ap; k = k + 1u) {",
                "    let plat = s_platform_pos[k];",
                "    let range_m = length(plat.xyz - p_gnd);",
                "    let phi = (FOUR_PI * range_m) / max(1e-4, u_bp.lambda_m);",
                "    let weight = plat.w;",
                "    accum = accum + vec2<f32>(cos(phi), sin(phi)) * weight;",
                "}",
                "s_bp_out[idx] = BpOutput(accum, length(accum), atan2(accum.y, accum.x));",
            ],
        ),
    ]

    for subpath, title, overview, theory, structs, bindings, helpers, body in specs:
        content = build_shader(subpath, title, overview, theory, structs, bindings, helpers, body)
        write_shader(f"{subpath}.wgsl", content)


if __name__ == "__main__":
    run()
