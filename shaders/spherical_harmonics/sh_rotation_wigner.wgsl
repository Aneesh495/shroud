// Shroud :: Wigner D-Matrix Rotation for Real Spherical Harmonics
// Rotates L0-L2 SH coefficients under an arbitrary 3x3 rotation matrix R.

struct ShCoeffsL2 {
    c: array<vec3<f32>, 9>,
};

struct ShRotationParams {
    rot_matrix: mat3x3<f32>,
};

fn rotate_sh_l1(r: mat3x3<f32>, c_in: ptr<function, array<vec3<f32>, 9>>, c_out: ptr<function, array<vec3<f32>, 9>>) {
    // L=1 corresponds to y, z, x coordinates: rotated directly by R
    for (var ch = 0u; ch < 3u; ch = ch + 1u) {
        let v = vec3<f32>((*c_in)[3][ch], (*c_in)[1][ch], (*c_in)[2][ch]); // (x, y, z)
        let v_rot = r * v;
        (*c_out)[3][ch] = v_rot.x;
        (*c_out)[1][ch] = v_rot.y;
        (*c_out)[2][ch] = v_rot.z;
    }
}

@group(0) @binding(0) var<uniform> u_rot_params: ShRotationParams;
@group(0) @binding(1) var<storage, read> s_sh_in: array<ShCoeffsL2>;
@group(0) @binding(2) var<storage, read_write> s_sh_out: array<ShCoeffsL2>;

@compute @workgroup_size(64, 1, 1)
fn cs_rotate_sh_coefficients(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_sh_in)) {
        return;
    }
    var in_c = s_sh_in[idx].c;
    var out_c = in_c;
    // L=0 is rotationally invariant
    out_c[0] = in_c[0];
    rotate_sh_l1(u_rot_params.rot_matrix, &in_c, &out_c);
    var res: ShCoeffsL2;
    res.c = out_c;
    s_sh_out[idx] = res;
}
