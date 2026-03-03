// Shroud :: Ratio Tracking Transmittance Estimator
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
