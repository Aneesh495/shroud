// Shroud :: Henyey-Greenstein Phase Function & Importance Sampling
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
