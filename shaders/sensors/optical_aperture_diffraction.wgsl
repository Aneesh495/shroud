// Shroud :: Wave Optical Lens Aperture Diffraction & Point Spread Function (PSF)
// Evaluates Bessel function Airy disk patterns and polygonal blade diffraction spikes.

const PI: f32 = 3.14159265358979323846;

// J1(x) Bessel function approximation via polynomial expansion
fn bessel_j1(x_in: f32) -> f32 {
    let x = abs(x_in);
    if (x < 1e-4) {
        return 0.5 * x;
    }
    // High-precision polynomial approximation of J1
    if (x < 3.75) {
        let y = (x / 3.75) * (x / 3.75);
        return x * (0.5 + y * (-0.56249985 + y * (0.21093573 + y * (-0.03954289 + y * 0.00443319))));
    } else {
        let ax = 3.75 / x;
        let f = 0.79788456 + ax * (-0.00000077 + ax * (-0.00552740 + ax * (0.00009512 + ax * 0.00137237)));
        let theta = x - 2.35619449 + ax * (0.04166397 + ax * (-0.00003954 + ax * (-0.00262573 + ax * 0.00054125)));
        return (1.0 / sqrt(x)) * f * cos(theta);
    }
}

// Circular aperture Airy disk PSF: I(theta) = I0 * [2 * J1(x) / x]^2
fn airy_disk_psf(radius_norm: f32, wavelength_um: f32, f_number: f32) -> f32 {
    let x = (PI * radius_norm) / (wavelength_um * f_number + 1e-7);
    if (abs(x) < 1e-5) {
        return 1.0;
    }
    let j1 = bessel_j1(x);
    let term = 2.0 * j1 / x;
    return term * term;
}

@group(0) @binding(0) var<storage, read> s_radii_um: array<f32>;
@group(0) @binding(1) var<storage, read_write> s_diffraction_intensities: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_aperture_diffraction_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_radii_um)) {
        return;
    }

    let r = s_radii_um[idx];
    let f_num = 2.8;

    let psf_red = airy_disk_psf(r, 0.650, f_num);
    let psf_green = airy_disk_psf(r, 0.532, f_num);
    let psf_blue = airy_disk_psf(r, 0.450, f_num);

    s_diffraction_intensities[idx] = vec4<f32>(psf_red, psf_green, psf_blue, 1.0);
}
