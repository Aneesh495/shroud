// Shroud :: Hero Wavelength Spectral Volume Tracking
// Tracks chromatic extinction across 4 correlated spectral wavelength bands.

struct HeroSpectralParams {
    wavelengths: vec4<f32>,     // e.g. 650, 532, 450, 905 nm
    sigma_t_spectrum: vec4<f32>,
    albedo_spectrum: vec4<f32>,
};

struct SpectralRadianceOutput {
    radiance: vec4<f32>,
    optical_depth: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u_spectral_vol: HeroSpectralParams;
@group(0) @binding(1) var<storage, read> s_path_lengths: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_spectral_out: array<SpectralRadianceOutput>;

@compute @workgroup_size(64, 1, 1)
fn cs_hero_spectral_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_path_lengths)) {
        return;
    }

    let dist = s_path_lengths[idx];
    let tau = u_spectral_vol.sigma_t_spectrum * dist;
    let transmittance = exp(-tau);

    let in_scatter = (vec4<f32>(1.0) - transmittance) * u_spectral_vol.albedo_spectrum;

    var out_rec: SpectralRadianceOutput;
    out_rec.radiance = in_scatter;
    out_rec.optical_depth = tau;
    s_spectral_out[idx] = out_rec;
}
