// Shroud :: Common Physical & Optical Constants
// Module: shaders/common/constants.wgsl
// CODATA 2022 SI fundamental constants and standard CIE colorimetric anchors.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const INV_PI: f32 = 0.31830988618379067154;
const INV_FOUR_PI: f32 = 0.07957747154594766788;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_CONSTANT: f32 = 6.62607015e-34;
const BOLTZMANN_CONSTANT: f32 = 1.380649e-23;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;

@group(0) @binding(0) var<storage, read_write> s_constants_verify: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&s_constants_verify)) {
        s_constants_verify[idx] = vec4<f32>(PI, TWO_PI, INV_PI, SPEED_OF_LIGHT);
    }
}
