// Shroud :: Cooley-Tukey Radix-2 Fast Fourier Transform (FFT)
// In-place butterfly operation across complex signals for FMCW spectrum analysis.

const PI: f32 = 3.14159265358979323846;

struct Complex {
    real: f32,
    imag: f32,
};

fn complex_mul(a: Complex, b: Complex) -> Complex {
    return Complex(
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    );
}

struct FftStageParams {
    stage_stride: u32,
    sub_sequence_len: u32,
    num_samples: u32,
    is_inverse: u32,
};

@group(0) @binding(0) var<uniform> u_fft: FftStageParams;
@group(0) @binding(1) var<storage, read_write> s_spectrum: array<Complex>;

@compute @workgroup_size(64, 1, 1)
fn cs_fft_butterfly_stage(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let half_len = u_fft.stage_stride;
    let group_size = half_len * 2u;

    if (idx >= u_fft.num_samples / 2u) {
        return;
    }

    let group = idx / half_len;
    let pair_offset = idx % half_len;

    let idx_even = group * group_size + pair_offset;
    let idx_odd = idx_even + half_len;

    let angle = -2.0 * PI * f32(pair_offset) / f32(group_size);
    let twiddle = Complex(cos(angle), sin(angle));

    let even_val = s_spectrum[idx_even];
    let odd_val = s_spectrum[idx_odd];

    let twiddled_odd = complex_mul(odd_val, twiddle);

    s_spectrum[idx_even] = Complex(even_val.real + twiddled_odd.real, even_val.imag + twiddled_odd.imag);
    s_spectrum[idx_odd] = Complex(even_val.real - twiddled_odd.real, even_val.imag - twiddled_odd.imag);
}
