// Shroud :: Numerical Throughput & Convergence Verification Script
// Script: scripts/benchmarks/bench_03_hg_fog_dense.rhai

fn run_phase_quadrature_convergence(n_steps, g_asym) {
    let d_mu = 2.0 / n_steps;
    let total = 0.0;
    let i = 0;
    while i < n_steps {
        let mu = -1.0 + (i + 0.5) * d_mu;
        total += 2.0 * 3.141592653589793 * eval_henyey_greenstein(mu, g_asym) * d_mu;
        i += 1;
    }
    assert_true(total > 0.90 && total < 1.10, "quadrature integral converged");
    total
}

fn run_ray_box_throughput(n_rays) {
    let bmin = vec3(-2.0, -2.0, 20.0);
    let bmax = vec3(2.0, 2.0, 25.0);
    let hits = 0;
    let i = 0;
    while i < n_rays {
        let offset = -1.5 + 3.0 * (i / n_rays);
        let r = ray(vec3(offset, 0.0, 0.0), vec3(0.0, 0.0, 1.0));
        if r.intersect_box(bmin, bmax) > 0.0 {
            hits += 1;
        }
        i += 1;
    }
    assert_true(hits == n_rays, "all central beams hit target AABB");
    hits
}

let q = run_phase_quadrature_convergence(128, 0.850);
let h = run_ray_box_throughput(64);
assert_true(validate_wgsl("shaders/aerosol_models/aerosol_radiation_ground_fog.wgsl"), "benchmark shader A valid");
assert_true(validate_wgsl("shaders/aerosol_models/aerosol_advection_sea_fog.wgsl"), "benchmark shader B valid");

#{"benchmark": "hg_fog_dense", "quadrature": q, "ray_hits": h}
