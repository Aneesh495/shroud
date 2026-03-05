// Shroud :: Core ReSTIR Reservoir Structures and Resampling Primitives
// Implements Weighted Reservoir Sampling (WRS) algorithms for online sample streams.

struct LightSample {
    position: vec3<f32>,
    radiance: vec3<f32>,
    normal: vec3<f32>,
    light_id: u32,
};

struct DiReservoir {
    sample_val: LightSample,
    weight_sum: f32, // w_sum
    m_count: f32,    // M
    w_weight: f32,   // W
    padding: f32,
};

fn reservoir_init() -> DiReservoir {
    var r: DiReservoir;
    r.weight_sum = 0.0;
    r.m_count = 0.0;
    r.w_weight = 0.0;
    r.padding = 0.0;
    return r;
}

// Updates reservoir with a new candidate sample using pseudo-random draw
fn reservoir_update(
    r: ptr<function, DiReservoir>,
    candidate: LightSample,
    weight: f32,
    random_u: f32
) -> bool {
    (*r).weight_sum += weight;
    (*r).m_count += 1.0;

    if (random_u * (*r).weight_sum < weight) {
        (*r).sample_val = candidate;
        return true;
    }
    return false;
}

// Finalizes reservoir weight W = w_sum / (M * p_hat)
fn reservoir_finalize(r: ptr<function, DiReservoir>, target_p_hat: f32) {
    if ((*r).m_count > 0.0 && target_p_hat > 1e-7) {
        (*r).w_weight = (*r).weight_sum / ((*r).m_count * target_p_hat);
    } else {
        (*r).w_weight = 0.0;
    }
}

@group(0) @binding(0) var<storage, read> s_candidates: array<LightSample>;
@group(0) @binding(1) var<storage, read_write> s_reservoirs_out: array<DiReservoir>;

@compute @workgroup_size(64, 1, 1)
fn cs_reservoir_primitive_test(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_candidates)) {
        return;
    }

    let cand = s_candidates[idx];
    var r = reservoir_init();
    reservoir_update(&r, cand, 1.5, 0.2);
    reservoir_finalize(&r, 1.0);

    s_reservoirs_out[idx] = r;
}
