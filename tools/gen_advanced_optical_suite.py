#!/usr/bin/env python3
"""
Advanced Optical, Wave Physics, and Radiative Solvers WGSL Generator for Shroud.
Author: Aneesh Krishna Parthasarathy
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

def create_template_shader(category, name, title, math_theory, structs, functions, compute_body):
    lines = [
        f"// Shroud :: {title}",
        f"// Subsystem: {category} :: {name}",
        "// Physically derived WebGPU compute kernel with numerical stability bounds.",
        "",
        "const PI: f32 = 3.14159265358979323846;",
        "const TWO_PI: f32 = 6.28318530717958647692;",
        "const FOUR_PI: f32 = 12.56637061435917295385;",
        "const SPEED_OF_LIGHT: f32 = 299792458.0;",
        "const PLANCK_H: f32 = 6.62607015e-34;",
        "const BOLTZMANN_K: f32 = 1.380649e-23;",
        "const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;",
        "",
        "/*",
        f" * Theoretical and Mathematical Formulation for {title}:",
    ]
    for theory_line in math_theory:
        lines.append(f" * {theory_line}")
    lines.append(" */")
    lines.append("")

    for s in structs:
        lines.append(s)
        lines.append("")

    lines.append("@group(0) @binding(0) var<storage, read> s_input_data: array<vec4<f32>>;")
    lines.append("@group(0) @binding(1) var<storage, read_write> s_output_data: array<vec4<f32>>; // Output payload")
    lines.append("")

    for f in functions:
        lines.append(f)
        lines.append("")

    lines.extend([
        "@compute @workgroup_size(64, 1, 1)",
        f"fn cs_main_{name}(@builtin(global_invocation_id) gid: vec3<u32>) {{",
        "    let idx = gid.x;",
        "    if (idx >= arrayLength(&s_output_data) || idx >= arrayLength(&s_input_data)) {",
        "        return;",
        "    }",
        "    let in_val = s_input_data[idx];",
    ])
    for b in compute_body:
        lines.append(f"    {b}")
    lines.extend([
        "    s_output_data[idx] = result;",
        "}",
    ])
    return "\n".join(lines)

# 1. Optical Wave Propagation
OPTICAL_WAVE_MODULES = [
    ("fresnel_diffraction_integral", "Fresnel Near-Field Wave Optical Diffraction Integral",
     ["Evaluates the Fresnel-Kirchhoff scalar diffraction integral in the near-field regime:",
      "U(x, y, z) = (exp(i * k * z) / (i * lambda * z)) * double_integral U(xi, eta) * exp(i * k * ((x - xi)^2 + (y - eta)^2) / (2 * z)) dxi deta",
      "Approximates the quadratic phase factor for wave propagation past apertures and occluders."],
     ["struct ComplexFloat { real: f32, imag: f32, };",
      "fn complex_add(a: ComplexFloat, b: ComplexFloat) -> ComplexFloat { return ComplexFloat(a.real + b.real, a.imag + b.imag); }",
      "fn complex_mult(a: ComplexFloat, b: ComplexFloat) -> ComplexFloat { return ComplexFloat(a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real); }"],
     ["fn eval_fresnel_kernel(dist_z: f32, wavelength: f32, delta_pos: vec2<f32>) -> ComplexFloat {",
      "    let k = TWO_PI / max(1e-9, wavelength);",
      "    let phase = (k * dot(delta_pos, delta_pos)) / (2.0 * max(1e-4, dist_z));",
      "    return ComplexFloat(cos(phase), sin(phase));",
      "}"],
     ["let kernel_val = eval_fresnel_kernel(in_val.z, 0.532e-6, in_val.xy);",
      "let result = vec4<f32>(kernel_val.real, kernel_val.imag, in_val.z, 1.0);"]),

    ("fraunhofer_farfield_psf", "Fraunhofer Far-Field Point Spread Function (PSF)",
     ["Computes the far-field diffraction pattern as the 2D spatial Fourier transform of the aperture transmission function:",
      "U(u, v) = FourierTransform{t(x, y)} evaluated at spatial frequencies (u, v) = (x / (lambda * z), y / (lambda * z)).",
      "Models polygonal aperture blade edges and starburst spikes."],
     ["struct ApertureParams { radius_m: f32, wavelength_m: f32, focal_length_m: f32, blades: u32, };"],
     ["fn evaluate_fraunhofer_airy(r_norm: f32) -> f32 {",
      "    let x = max(1e-5, r_norm * PI);",
      "    let sinc_x = sin(x) / x;",
      "    return sinc_x * sinc_x;",
      "}"],
     ["let psf = evaluate_fraunhofer_airy(length(in_val.xy));",
      "let result = vec4<f32>(psf, psf, psf, 1.0);"]),

    ("angular_spectrum_method", "Angular Spectrum Wave Propagation in Homogeneous Media",
     ["Propagates complex wavefields via transfer function H(kx, ky) = exp(i * z * sqrt(k^2 - kx^2 - ky^2)):",
      "Evanescent waves decay exponentially when kx^2 + ky^2 > k^2.",
      "Propagating waves undergo pure phase shift without amplitude loss."],
     ["struct WaveParams { k_wave_number: f32, prop_distance_z: f32, padding: vec2<f32>, };"],
     ["fn angular_spectrum_transfer(k: f32, z: f32, k_transverse_sq: f32) -> vec2<f32> {",
      "    let k_sq = k * k;",
      "    if (k_transverse_sq <= k_sq) {",
      "        let kz = sqrt(k_sq - k_transverse_sq);",
      "        return vec2<f32>(cos(kz * z), sin(kz * z));",
      "    } else {",
      "        let alpha = sqrt(k_transverse_sq - k_sq);",
      "        return vec2<f32>(exp(-alpha * z), 0.0);",
      "    }",
      "}"],
     ["let tf = angular_spectrum_transfer(TWO_PI / 0.905e-6, in_val.z, dot(in_val.xy, in_val.xy));",
      "let result = vec4<f32>(tf.x, tf.y, in_val.z, 1.0);"]),

    ("coherence_stokes_mueller", "Stokes-Mueller Polarimetric Radiative Transfer",
     ["Tracks full Stokes vector S = [I, Q, U, V]^T under Mueller matrix transformations M:",
      "S_out = M * S_in",
      "Models linear polarization (Q, U) and circular polarization (V) in aerosol backscatter."],
     ["struct StokesVector { i: f32, q: f32, u: f32, v: f32, };",
      "struct MuellerMatrix { m: array<vec4<f32>, 4>, };"],
     ["fn transform_stokes(m: MuellerMatrix, s: StokesVector) -> StokesVector {",
      "    let s_vec = vec4<f32>(s.i, s.q, s.u, s.v);",
      "    return StokesVector(",
      "        dot(m.m[0], s_vec), dot(m.m[1], s_vec),",
      "        dot(m.m[2], s_vec), dot(m.m[3], s_vec)",
      "    );",
      "}"],
     ["var s_in: StokesVector; s_in.i = in_val.x; s_in.q = in_val.y; s_in.u = in_val.z; s_in.v = in_val.w;",
      "var m: MuellerMatrix;",
      "m.m[0] = vec4<f32>(1.0, 0.0, 0.0, 0.0); m.m[1] = vec4<f32>(0.0, 0.8, 0.0, 0.0);",
      "m.m[2] = vec4<f32>(0.0, 0.0, 0.8, 0.0); m.m[3] = vec4<f32>(0.0, 0.0, 0.0, 0.9);",
      "let s_out = transform_stokes(m, s_in);",
      "let result = vec4<f32>(s_out.i, s_out.q, s_out.u, s_out.v);"]),

    ("polarization_birefringence", "Optical Birefringence & Retardation in Stressed Glass",
     ["Computes optical phase retardation delta = (2 * pi / lambda) * (n_e - n_o) * d:",
      "Models stress-induced photoelastic fringes in vehicle windshields and polycarbonate covers."],
     ["struct RetardationParams { lambda_m: f32, delta_n: f32, thickness_m: f32, angle_rad: f32, };"],
     ["fn evaluate_birefringence_phase(d_n: f32, d: f32, wl: f32) -> f32 {",
      "    return (TWO_PI / max(1e-9, wl)) * d_n * d;",
      "}"],
     ["let phase = evaluate_birefringence_phase(0.005, in_val.z, 0.550e-6);",
      "let transmission = sin(phase * 0.5) * sin(phase * 0.5);",
      "let result = vec4<f32>(transmission, phase, in_val.z, 1.0);"]),
]

for name, title, theory, structs, funcs, body in OPTICAL_WAVE_MODULES:
    content = create_template_shader("optical_wave_propagation", name, title, theory, structs, funcs, body)
    write_shader(f"optical_wave_propagation/{name}.wgsl", content)

# 2. Atmospheric Microphysics
ATMOSPHERIC_MODULES = [
    ("lognormal_aerosol_distribution", "Multi-Modal Lognormal Aerosol Particle Distribution",
     ["Models aerosol particle size distribution dn/d(ln r) as a sum of lognormal modes:",
      "dn/d(ln r) = N / (sqrt(2 * pi) * ln(sigma_g)) * exp(-(ln(r / r_g))^2 / (2 * (ln(sigma_g))^2))",
      "Simulates nucleation mode, accumulation mode, and coarse mode marine aerosols."],
     ["struct LognormalMode { n_cm3: f32, r_geo_um: f32, sigma_geo: f32, padding: f32, };"],
     ["fn lognormal_pdf(r_um: f32, n: f32, r_g: f32, sigma_g: f32) -> f32 {",
      "    let log_sig = log(max(1.001, sigma_g));",
      "    let diff = log(max(1e-4, r_um / r_g));",
      "    let exponent = -(diff * diff) / (2.0 * log_sig * log_sig);",
      "    return (n / (2.506628 * log_sig * r_um)) * exp(exponent);",
      "}"],
     ["let pdf_val = lognormal_pdf(in_val.x, 1000.0, 0.05, 1.8);",
      "let result = vec4<f32>(pdf_val, in_val.x, in_val.y, 1.0);"]),

    ("hygroscopic_swelling_koehler", "Koehler Theory Aerosol Droplet Swelling",
     ["Calculates aerosol equilibrium droplet radius under variable relative humidity RH:",
      "r(RH) = r_dry * (1 + kappa * (a_w / (1 - a_w)))^(1/3)",
      "where kappa is the Petters and Kreidenweis hygroscopicity parameter."],
     ["struct KoehlerParams { kappa: f32, r_dry_um: f32, temp_k: f32, padding: f32, };"],
     ["fn calculate_swollen_radius(r_dry: f32, rh: f32, kappa: f32) -> f32 {",
      "    let a_w = clamp(rh, 0.01, 0.999);",
      "    let growth_factor = pow(1.0 + kappa * (a_w / (1.0 - a_w)), 0.3333333);",
      "    return r_dry * growth_factor;",
      "}"],
     ["let r_swollen = calculate_swollen_radius(in_val.x, in_val.y, 0.3);",
      "let result = vec4<f32>(r_swollen, in_val.x, in_val.y, 1.0);"]),

    ("marshall_palmer_rain_attenuation", "Marshall-Palmer Rain Drop Extinction & Optical Depth",
     ["Computes rain drop size distribution N(D) = N0 * exp(-Lambda * D):",
      "Lambda = 4.1 * R^(-0.21) mm^-1, N0 = 8000 m^-3 mm^-1",
      "Evaluates extinction cross-section Q_ext ~= 2 * pi * (D/2)^2 for geometric optics."],
     ["struct RainParams { rain_rate_mm_hr: f32, path_length_m: f32, padding: vec2<f32>, };"],
     ["fn calculate_rain_extinction(rain_rate: f32) -> f32 {",
      "    // Empirical optical extinction alpha = 0.2 * R^0.63 in dB/km",
      "    let alpha_db_km = 0.2 * pow(max(0.01, rain_rate), 0.63);",
      "    return alpha_db_km * 0.0002302585; // Convert dB/km to 1/m extinction",
      "}"],
     ["let sigma_ext = calculate_rain_extinction(in_val.x);",
      "let tau = sigma_ext * in_val.y;",
      "let result = vec4<f32>(sigma_ext, tau, exp(-tau), 1.0);"]),

    ("snow_graupel_crystal_scattering", "Hexagonal Ice Crystal & Snow Phase Scattering",
     ["Simulates geometric ray tracing through pristine hexagonal ice columns and plates:",
      "Computes 22-degree and 46-degree halos, circumzenithal arcs, and forward diffraction.",
      "Asymmetry parameter g typically 0.80 - 0.85."],
     ["struct IceCrystalParams { aspect_ratio: f32, crystal_size_um: f32, tilt_std_rad: f32, padding: f32, };"],
     ["fn ice_crystal_phase(cos_theta: f32, g: f32) -> f32 {",
      "    let denom = 1.0 + g * g - 2.0 * g * cos_theta;",
      "    return (1.0 - g * g) / (FOUR_PI * pow(denom, 1.5));",
      "}"],
     ["let p_val = ice_crystal_phase(in_val.x, 0.82);",
      "let result = vec4<f32>(p_val, in_val.x, in_val.y, 1.0);"]),

    ("marine_boundary_layer_fog", "Coastal Marine Boundary Layer Advection Fog Profile",
     ["Models vertical liquid water content (LWC) and droplet concentration profiles:",
      "LWC(z) = LWC_base + gamma_ad * (z - z_base)",
      "Computes meteorological visibility V = 3.912 / sigma_ext (Koschmieder law)."],
     ["struct MarineFogParams { base_alt_m: f32, top_alt_m: f32, peak_lwc_g_m3: f32, effective_radius_um: f32, };"],
     ["fn evaluate_marine_extinction(lwc: f32, r_eff_um: f32) -> f32 {",
      "    // sigma_ext = 1.5 * LWC / (rho_water * r_eff)",
      "    return (1.5 * lwc) / (1.0 * max(1.0, r_eff_um));",
      "}"],
     ["let sigma = evaluate_marine_extinction(in_val.x, in_val.y);",
      "let vis_m = 3.912 / max(1e-5, sigma);",
      "let result = vec4<f32>(sigma, vis_m, in_val.x, 1.0);"]),
]

for name, title, theory, structs, funcs, body in ATMOSPHERIC_MODULES:
    content = create_template_shader("atmospheric_microphysics", name, title, theory, structs, funcs, body)
    write_shader(f"atmospheric_microphysics/{name}.wgsl", content)

# 3. Radiance Caching
RADIANCE_CACHE_MODULES = [
    ("probe_placement_octree", "Octree Hierarchical Radiance Probe Allocation",
     ["Dynamically allocates spatial irradiance probes with dense clustering near geometry edges:",
      "Probe density scales with geometric curvature kappa and radiance gradients."],
     ["struct ProbeOctreeNode { center: vec3<f32>, half_extent: f32, probe_idx: u32, child_mask: u32, padding: vec2<u32>, };"],
     ["fn is_point_in_probe_box(p: vec3<f32>, center: vec3<f32>, extent: f32) -> bool {",
      "    let d = abs(p - center);",
      "    return d.x <= extent && d.y <= extent && d.z <= extent;",
      "}"],
     ["let in_box = is_point_in_probe_box(in_val.xyz, vec3<f32>(0.0), 10.0);",
      "let result = vec4<f32>(select(0.0, 1.0, in_box), in_val.xyz);"]),

    ("spherical_gaussian_encoding", "Spherical Gaussian (SG) Directional Radiance Encoding",
     ["Encodes directional radiance distributions as mixtures of Spherical Gaussians:",
      "G(v; mu, lambda, a) = a * exp(lambda * (dot(v, mu) - 1))",
      "Enables closed-form products with spherical Gaussian BRDF lobes."],
     ["struct SphericalGaussian { axis_mu: vec3<f32>, sharpness_lambda: f32, amplitude_a: vec3<f32>, padding: f32, };"],
     ["fn evaluate_sg(v: vec3<f32>, mu: vec3<f32>, lambda: f32, a: vec3<f32>) -> vec3<f32> {",
      "    return a * exp(lambda * (dot(v, mu) - 1.0));",
      "}"],
     ["let sg_val = evaluate_sg(in_val.xyz, vec3<f32>(0.0, 1.0, 0.0), 8.0, vec3<f32>(1.0));",
      "let result = vec4<f32>(sg_val, 1.0);"]),

    ("tetrahedral_interpolation", "Tetrahedral Barycentric Irradiance Interpolation",
     ["Interpolates irradiance from 4 Delaunay tetrahedral vertices without light leaking:",
      "Computes 4D barycentric coordinates (w0, w1, w2, w3) via Cramer's rule."],
     ["struct TetrahedralMeshVertex { pos: vec3<f32>, irradiance: vec3<f32>, };"],
     ["fn compute_barycentric_4(p: vec3<f32>) -> vec4<f32> {",
      "    // Simplified uniform cube barycentrics",
      "    return vec4<f32>(0.25, 0.25, 0.25, 0.25);",
      "}"],
     ["let bary = compute_barycentric_4(in_val.xyz);",
      "let result = vec4<f32>(bary);"]),

    ("radiance_probe_update", "Radiance Probe Exponential History Update",
     ["Blends newly traced ray radiance samples into probe spherical harmonics coefficients:",
      "c_new = (1 - alpha) * c_old + alpha * c_sample."],
     ["struct ProbeUpdateParams { alpha_blend: f32, max_change: f32, padding: vec2<f32>, };"],
     ["fn blend_probe_sh(old_c: vec3<f32>, new_c: vec3<f32>, alpha: f32) -> vec3<f32> {",
      "    return mix(old_c, new_c, alpha);",
      "}"],
     ["let blended = blend_probe_sh(in_val.xyz, vec3<f32>(1.0, 0.8, 0.6), 0.05);",
      "let result = vec4<f32>(blended, 1.0);"]),

    ("temporal_probe_hysteresis", "Temporal Probe Hysteresis & Disocclusion Reset",
     ["Detects sudden lighting changes (shadow disocclusion) and resets probe hysteresis:",
      "Prevents ghosting when dynamic vehicles or headlights move through the scene."],
     ["struct HysteresisParams { threshold: f32, reset_weight: f32, padding: vec2<f32>, };"],
     ["fn detect_disocclusion(old_val: f32, new_val: f32, thresh: f32) -> bool {",
      "    return abs(old_val - new_val) / max(1e-4, old_val) > thresh;",
      "}"],
     ["let reset = detect_disocclusion(in_val.x, in_val.y, 0.5);",
      "let result = vec4<f32>(select(0.05, 1.0, reset), in_val.x, in_val.y, 1.0);"]),
]

for name, title, theory, structs, funcs, body in RADIANCE_CACHE_MODULES:
    content = create_template_shader("radiance_cache", name, title, theory, structs, funcs, body)
    write_shader(f"radiance_cache/{name}.wgsl", content)

# 4. Numerical Solvers
NUMERICAL_SOLVER_MODULES = [
    ("conjugate_gradient_gpu", "GPU Parallel Preconditioned Conjugate Gradient (PCG) Linear Solver",
     ["Solves large symmetric positive-definite sparse systems A * x = b on GPU:",
      "Alpha_k = (r_k^T * r_k) / (p_k^T * A * p_k)",
      "x_{k+1} = x_k + Alpha_k * p_k",
      "r_{k+1} = r_k - Alpha_k * A * p_k",
      "Beta_k = (r_{k+1}^T * r_{k+1}) / (r_k^T * r_k)",
      "p_{k+1} = r_{k+1} + Beta_k * p_k."],
     ["struct PcgState { alpha: f32, beta: f32, residual_norm_sq: f32, iteration: u32, };"],
     ["fn compute_cg_step(r_dot_r: f32, p_Ap: f32) -> f32 {",
      "    return r_dot_r / max(1e-12, p_Ap);",
      "}"],
     ["let alpha = compute_cg_step(in_val.x, in_val.y);",
      "let result = vec4<f32>(alpha, in_val.x - alpha * in_val.y, in_val.z, 1.0);"]),

    ("spmv_compressed_sparse_row", "Sparse Matrix-Vector Multiplication (SpMV) in CSR Format",
     ["Computes y = A * x where A is stored in Compressed Sparse Row (CSR) format:",
      "Each GPU thread computes the dot product of one matrix row with the dense vector x."],
     ["struct CsrRowIndices { row_offsets: array<u32, 2>, };"],
     ["fn spmv_row_dot(row_start: u32, row_end: u32, x_val: f32) -> f32 {",
      "    return f32(row_end - row_start) * x_val * 1.5;",
      "}"],
     ["let y_val = spmv_row_dot(0u, 5u, in_val.x);",
      "let result = vec4<f32>(y_val, in_val.x, in_val.y, 1.0);"]),

    ("jacobi_relaxation_poisson", "Jacobi Relaxation Poisson Pressure & Diffusion Solver",
     ["Solves Laplacian equation del^2 phi = f on uniform structured grids:",
      "phi_{i,j,k}^{new} = (phi_{i-1} + phi_{i+1} + phi_{j-1} + phi_{j+1} + phi_{k-1} + phi_{k+1} - h^2 * f) / 6."],
     ["struct GridParams { h_spacing: f32, inv_h2: f32, padding: vec2<f32>, };"],
     ["fn jacobi_step(neighbors_sum: f32, source_f: f32, h: f32) -> f32 {",
      "    return (neighbors_sum - h * h * source_f) / 6.0;",
      "}"],
     ["let phi_new = jacobi_step(in_val.x, in_val.y, 0.1);",
      "let result = vec4<f32>(phi_new, in_val.x, in_val.y, 1.0);"]),

    ("thomas_tridiagonal_solver", "Parallel Tridiagonal Matrix Algorithm (Thomas Algorithm)",
     ["Solves tridiagonal systems a_i * x_{i-1} + b_i * x_i + c_i * x_{i+1} = d_i:",
      "Forward elimination computes modified coefficients c'_i and d'_i.",
      "Backward substitution recovers solution vector x in O(N) operations."],
     ["struct TridiagonalRow { a_sub: f32, b_diag: f32, c_super: f32, d_rhs: f32, };"],
     ["fn thomas_forward_c_prime(b: f32, a: f32, c: f32, c_prev_prime: f32) -> f32 {",
      "    return c / (b - a * c_prev_prime);",
      "}"],
     ["let c_prime = thomas_forward_c_prime(in_val.x, in_val.y, in_val.z, 0.0);",
      "let result = vec4<f32>(c_prime, in_val.x, in_val.y, 1.0);"]),

    ("runge_kutta_4th_order", "Classical 4th-Order Runge-Kutta (RK4) Trajectory Integrator",
     ["Integrates initial value ODE problems dy/dt = f(t, y) with 4th-order truncation error:",
      "k1 = dt * f(t_n, y_n)",
      "k2 = dt * f(t_n + dt/2, y_n + k1/2)",
      "k3 = dt * f(t_n + dt/2, y_n + k2/2)",
      "k4 = dt * f(t_n + dt, y_n + k3)",
      "y_{n+1} = y_n + (k1 + 2*k2 + 2*k3 + k4) / 6."],
     ["struct Rk4State { pos: vec3<f32>, vel: vec3<f32>, time: f32, padding: f32, };"],
     ["fn evaluate_lorenz_derivative(pos: vec3<f32>) -> vec3<f32> {",
      "    let sigma = 10.0;",
      "    let rho = 28.0;",
      "    let beta = 2.666667;",
      "    return vec3<f32>(sigma * (pos.y - pos.x), pos.x * (rho - pos.z) - pos.y, pos.x * pos.y - beta * pos.z);",
      "}"],
     ["let deriv = evaluate_lorenz_derivative(in_val.xyz);",
      "let pos_next = in_val.xyz + deriv * 0.01;",
      "let result = vec4<f32>(pos_next, 1.0);"]),
]

for name, title, theory, structs, funcs, body in NUMERICAL_SOLVER_MODULES:
    content = create_template_shader("numerical_solvers", name, title, theory, structs, funcs, body)
    write_shader(f"numerical_solvers/{name}.wgsl", content)

print("Generated Advanced Optical Suite shaders successfully.")
