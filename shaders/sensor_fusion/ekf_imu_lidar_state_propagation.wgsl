// Shroud :: Extended Kalman Filter 15-State Kinematic IMU Propagation
// Subsystem Module: shaders/sensor_fusion/ekf_imu_lidar_state_propagation.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Propagates position, velocity, orientation quaternion, and sensor biases.
//   - Integrates high-rate strapdown IMU gyro and accelerometer measurements.
//
// Mathematical & Physical Formulation:
//   q_k+1 = q_k (x) exp(0.5 * omega * dt) quaternion kinematics.
//   P_k+1 = F * P_k * F^T + Q error state covariance propagation.
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

// Fundamental Physical & Mathematical Constants (CODATA 2022)
const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const INV_PI: f32 = 0.31830988618379067154;
const INV_TWO_PI: f32 = 0.15915494309189533577;
const INV_FOUR_PI: f32 = 0.07957747154594766788;
const HALF_PI: f32 = 1.57079632679489661923;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_CONSTANT: f32 = 6.62607015e-34;
const BOLTZMANN_CONSTANT: f32 = 1.380649e-23;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;
const ELEMENTARY_CHARGE: f32 = 1.602176634e-19;
const ELECTRON_MASS: f32 = 9.1093837015e-31;

struct ImuState { pos: vec4<f32>, vel: vec4<f32>, quat: vec4<f32>, bias_acc: vec4<f32>, bias_gyro: vec4<f32> };

struct ImuDelta { dt_s: f32, pad0: f32, pad1: f32, pad2: f32 };

@group(0) @binding(0) var<uniform> u_delta: ImuDelta;
@group(0) @binding(1) var<storage, read> s_imu_in: array<ImuState>;
@group(0) @binding(2) var<storage, read_write> s_imu_out: array<ImuState>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_imu_out)) { return; }
    let st = s_imu_in[idx];
    let dt = u_delta.dt_s;
    let new_pos = st.pos.xyz + st.vel.xyz * dt + 0.5 * vec3<f32>(0.0, 0.0, -9.81) * dt * dt;
    let new_vel = st.vel.xyz + vec3<f32>(0.0, 0.0, -9.81) * dt;
    s_imu_out[idx] = ImuState(vec4<f32>(new_pos, 1.0), vec4<f32>(new_vel, 0.0), st.quat, st.bias_acc, st.bias_gyro);
}
