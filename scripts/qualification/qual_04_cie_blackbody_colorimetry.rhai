// Shroud :: Analytical & Numerical Qualification Suite
// Script: scripts/qualification/qual_04_cie_blackbody_colorimetry.rhai
// Verifies physical conservation laws, geometric invariants, and numerical stability.

fn run_blackbody_wien_check(t_k) {
    let peak_nm = eval_blackbody_peak_nm(t_k);
    let expected_nm = 2.89777e6 / t_k;
    if expected_nm >= 420.0 && expected_nm <= 680.0 {
        assert_close(peak_nm, expected_nm, 25.0);
    } else if expected_nm < 400.0 {
        assert_close(peak_nm, 400.0, 15.0);
    }
    let rgb = eval_blackbody_rgb(t_k);
    assert_true(rgb.len() == 3, "RGB CIE conversion returns 3 channels");
    assert_true(rgb[0] >= 0.0 && rgb[1] >= 0.0 && rgb[2] >= 0.0, "non-negative tristimulus");

    // Check Stefan-Boltzmann monotonic scaling in LWIR window
    let lwir_lo = eval_lwir_radiance(250.0);
    let lwir_mid = eval_lwir_radiance(300.0);
    let lwir_hi = eval_lwir_radiance(360.0);
    assert_gt(lwir_mid, lwir_lo);
    assert_gt(lwir_hi, lwir_mid);
    peak_nm
}

fn run_brown_conrady_radial_monotonicity(k1_c, k2_c) {
    let p1 = 0.0012;
    let p2 = -0.0008;
    let k3 = 0.005;

    let origin = brown_conrady_distort(0.0, 0.0, k1_c, k2_c, p1, p2, k3);
    assert_true(abs(origin[0]) < 1e-4 && abs(origin[1]) < 1e-4, "center distortion vanishes");

    let radii = [0.05, 0.15, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0];
    let prev_r = -1.0;
    for r in radii {
        let uv = brown_conrady_distort(r, 0.0, k1_c, k2_c, 0.0, 0.0, 0.0);
        let r_out = sqrt(uv[0] * uv[0] + uv[1] * uv[1]);
        assert_gt(r_out, prev_r);
        prev_r = r_out;
    }
    prev_r
}

fn run_morton_z_curve_monotonicity() {
    let coords = [
        vec3(0.00, 0.00, 0.00),
        vec3(0.02, 0.02, 0.02),
        vec3(0.05, 0.05, 0.05),
        vec3(0.12, 0.14, 0.11),
        vec3(0.24, 0.22, 0.25),
        vec3(0.35, 0.30, 0.32),
        vec3(0.51, 0.52, 0.50),
        vec3(0.68, 0.65, 0.70),
        vec3(0.75, 0.72, 0.78),
        vec3(0.95, 0.96, 0.99),
    ];
    let prev = -1;
    for pt in coords {
        let code = morton_3d(pt);
        assert_true(code >= 0, "30-bit Morton code non-negative");
        assert_true(code > prev, "diagonal Z-order curve strictly increasing");
        prev = code;
    }
    prev
}

fn run_vector_algebra_identities() {
    let u = vec3(1.0 + 0.60, 2.0, -1.5);
    let v = vec3(-0.5, 3.2, 4.1);
    let w = vec3(2.1, -1.7, 0.8);

    // Orthogonality of cross product
    let c = u.cross(v);
    assert_close(c.dot(u), 0.0, 1e-5);
    assert_close(c.dot(v), 0.0, 1e-5);

    // Anti-commutativity of cross product
    let c_rev = v.cross(u);
    assert_close(c.x + c_rev.x, 0.0, 1e-5);
    assert_close(c.y + c_rev.y, 0.0, 1e-5);
    assert_close(c.z + c_rev.z, 0.0, 1e-5);

    // Unit normalization invariant
    let un = u.normalize();
    assert_close(un.length(), 1.0, 1e-6);

    // Vector triple product identity: u x (v x w) == v*(u.w) - w*(u.v)
    let lhs = u.cross(v.cross(w));
    let rhs = v * u.dot(w) - w * u.dot(v);
    assert_close(lhs.x, rhs.x, 1e-5);
    assert_close(lhs.y, rhs.y, 1e-5);
    assert_close(lhs.z, rhs.z, 1e-5);

    // Linear interpolation midpoint and endpoints
    let mid = u.lerp(v, 0.5);
    let expected = (u + v) * 0.5;
    assert_close(mid.distance(expected), 0.0, 1e-6);
    assert_close(u.lerp(v, 0.0).distance(u), 0.0, 1e-6);
    assert_close(u.lerp(v, 1.0).distance(v), 0.0, 1e-6);
    true
}

fn run_ray_aabb_slab_precision() {
    let bmin = vec3(-2.0, -2.0, 10.0);
    let bmax = vec3(2.0, 2.0, 14.0);
    let r_hit = ray(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0));
    let t = r_hit.intersect_box(bmin, bmax);
    assert_close(t, 10.0, 1e-5);

    // Grazing parallel edge ray
    let r_edge = ray(vec3(1.999, 0.0, 0.0), vec3(0.0, 0.0, 1.0));
    let t_edge = r_edge.intersect_box(bmin, bmax);
    assert_close(t_edge, 10.0, 1e-5);

    // Oblique diagonal ray through box center (0, 0, 12)
    let r_diag = ray(vec3(-6.0, -6.0, 6.0), vec3(1.0, 1.0, 1.0));
    let t_diag = r_diag.intersect_box(bmin, bmax);
    assert_gt(t_diag, 0.0);

    // Reverse ray must not intersect forward slab
    let r_back = ray(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, -1.0));
    assert_close(r_back.intersect_box(bmin, bmax), -1.0, 1e-6);
    t
}

fn run_atmosphere_altitude_profile() {
    let altitudes = [0.0, 500.0, 1000.0, 2500.0, 5000.0, 8000.0, 11000.0, 15000.0, 20000.0];
    let prev_rho = 2.0;
    for h in altitudes {
        let rho = us_std_density(h);
        assert_lt(rho, prev_rho);
        assert_gt(rho, 0.0);
        prev_rho = rho;
    }
    prev_rho
}

fn run_fmcw_dual_band_invariants() {
    let f_rad = fmcw_config(77.0e9, 1.0e9, 40.0e-6, 0.10);
    let f_opt = fmcw_config(193.4e12, 800.0e6, 20.0e-6, 0.25);
    let pairs = [[15.0, 22.0], [50.0, -12.5], [140.0, 0.0], [210.0, -35.0]];
    for p in pairs {
        let r = p[0];
        let v = p[1];
        let u1 = f_rad.beat_up(r, v);
        let d1 = f_rad.beat_down(r, v);
        assert_close(f_rad.solve_range(u1, d1), r, 0.05);
        assert_close(f_rad.solve_velocity(u1, d1), v, 0.05);

        let u2 = f_opt.beat_up(r, v);
        let d2 = f_opt.beat_down(r, v);
        assert_close(f_opt.solve_range(u2, d2), r, 0.05);
        assert_close(f_opt.solve_velocity(u2, d2), v, 0.05);
    }
    true
}

let peak = run_blackbody_wien_check(5778.0);
let lwir_test = eval_lwir_radiance(330.00);
assert_gt(lwir_test, 10.0);

let max_r = run_brown_conrady_radial_monotonicity(-0.0600, 0.0450);
let morton_max = run_morton_z_curve_monotonicity();
run_vector_algebra_identities();
let t_slab = run_ray_aabb_slab_precision();
let rho_20k = run_atmosphere_altitude_profile();
run_fmcw_dual_band_invariants();

assert_true(validate_wgsl("shaders/common/colorimetry.wgsl"), "shader A compiled cleanly");
assert_true(validate_wgsl("shaders/sensors/thermal_lwir_microbolometer.wgsl"), "shader B compiled cleanly");

#{"qual": "cie_blackbody_colorimetry", "solar_peak_nm": peak, "max_r": max_r, "morton": morton_max, "slab_t": t_slab, "rho_20km": rho_20k}
