// Shroud :: Sensor Pipeline Specification: Aeva Aeries II 4D FMCW Coherent Doppler LiDAR
// Model ID: sensor_aeva_aeries2_fmcw
// Physical optical transceiver parameters, ray emission firing, and receiver signal processing.

const PI: f32 = 3.14159265358979323846;
const SPEED_OF_LIGHT: f32 = 299792458.0;

const SENSOR_CHANNELS: u32 = 1u;
const SENSOR_WAVELENGTH_NM: f32 = 1550.0;
const SENSOR_OPERATING_RATE: f32 = 0.0;
const SENSOR_MAX_RANGE_M: f32 = 300.0;
const SENSOR_RANGE_RESOLUTION_M: f32 = 0.0100;
const SENSOR_BEAM_DIVERGENCE_RAD: f32 = 0.000800;

/*
 * Operational Specifications for Aeva Aeries II 4D FMCW Coherent Doppler LiDAR:
 * - Active Channels / Detector Pixels: 1
 * - Optical Operating Center Wavelength: 1550.0 nm
 * - Frame Rate / Rotational Velocity: 0.0
 * - Maximum Instrumented Unambiguous Range: 300.0 m
 * - Systematic Range Bin Resolution: 0.0100 m
 * - Transmitter Optical Beam Divergence: 0.000800 rad
 *
 * Radiometric Transfer Link Budget:
 * Received power P_rx is computed via the radar range equation for extended targets:
 * P_rx = P_tx * (rho / pi) * (A_rx / R^2) * eta_sys * exp(-2 * alpha * R)
 */

struct TransceiverConfig {
    tx_power_peak: f32,
    rx_aperture_area: f32,
    system_efficiency: f32,
    noise_floor_w: f32,
    sensor_position: vec4<f32>,
    sensor_orientation_quat: vec4<f32>,
};

struct TargetEchoReturn {
    range_measured_m: f32,
    received_power_w: f32,
    snr_db: f32,
    valid_detection: u32,
    doppler_radial_velocity_m_s: f32,
    channel_id: u32,
    padding: vec2<f32>,
};

@group(0) @binding(0) var<uniform> u_config: TransceiverConfig;
@group(0) @binding(1) var<storage, read> s_target_ranges: array<f32>;
@group(0) @binding(2) var<storage, read> s_target_reflectances: array<f32>;
@group(0) @binding(3) var<storage, read_write> s_echoes_out: array<TargetEchoReturn>;

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn calculate_link_budget(r: f32, refl: f32, p_tx: f32, a_rx: f32, eta: f32) -> f32 {
    if (r < 0.1) {
        return 0.0;
    }
    let geom = a_rx / (PI * r * r);
    let atm_extinction = exp(-0.004 * r); // Atmospheric attenuation in fog
    return p_tx * refl * geom * eta * atm_extinction * atm_extinction;
}

@compute @workgroup_size(64, 1, 1)
fn cs_pipeline_aeva_aeries2_fmcw(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_echoes_out) || idx >= arrayLength(&s_target_ranges)) {
        return;
    }
    let r_true = s_target_ranges[idx];
    let refl = s_target_reflectances[idx];

    let p_rx = calculate_link_budget(
        r_true,
        refl,
        u_config.tx_power_peak,
        u_config.rx_aperture_area,
        u_config.system_efficiency
    );

    let snr_linear = p_rx / max(1e-15, u_config.noise_floor_w);
    let snr_db = 10.0 * log(max(1e-3, snr_linear)) / log(10.0);

    var ret: TargetEchoReturn;
    ret.range_measured_m = r_true + 0.005 * (f32(idx % 7u) - 3.0); // Range jitter
    ret.received_power_w = p_rx;
    ret.snr_db = snr_db;
    ret.valid_detection = select(0u, 1u, snr_db > 6.0 && r_true <= SENSOR_MAX_RANGE_M);
    ret.doppler_radial_velocity_m_s = 0.0;
    ret.channel_id = idx % SENSOR_CHANNELS;
    ret.padding = vec2<f32>(0.0, 0.0);

    s_echoes_out[idx] = ret;
}
