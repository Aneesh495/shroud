# Radiative Transfer & Participating Media

Shroud models optical, infrared, and millimeter-wave propagation through heterogeneous participating media such as radiation fog, advection marine layers, blowing desert dust, wildfire biomass smoke, rain, and upper-atmospheric Rayleigh columns.

## Volume Radiative Transfer Equation (RTE)

Along a ray $\mathbf{x}(s) = \mathbf{x}_0 + s \boldsymbol{\omega}$ parameterized by distance $s$, the monochromatic spectral radiance $L(\mathbf{x}, \boldsymbol{\omega}, \lambda)$ satisfies:

$$
\frac{d L(\mathbf{x}, \boldsymbol{\omega})}{ds} = -\sigma_t(\mathbf{x}) L(\mathbf{x}, \boldsymbol{\omega}) + \sigma_a(\mathbf{x}) L_e(\mathbf{x}, \boldsymbol{\omega}) + \sigma_s(\mathbf{x}) \int_{\mathcal{S}^2} p(\mathbf{x}, \boldsymbol{\omega}' \cdot \boldsymbol{\omega}) L_i(\mathbf{x}, \boldsymbol{\omega}') \, d\boldsymbol{\omega}'
$$

where:
- $\sigma_a(\mathbf{x})$ is the absorption coefficient ($\text{m}^{-1}$),
- $\sigma_s(\mathbf{x})$ is the scattering coefficient ($\text{m}^{-1}$),
- $\sigma_t(\mathbf{x}) = \sigma_a(\mathbf{x}) + \sigma_s(\mathbf{x})$ is the total extinction coefficient ($\text{m}^{-1}$),
- $\Lambda = \sigma_s / \sigma_t$ is the single-scattering albedo,
- $p(\cos\theta)$ is the normalized angular phase function satisfying $\int_{\mathcal{S}^2} p(\boldsymbol{\omega}' \cdot \boldsymbol{\omega}) \, d\boldsymbol{\omega}' = 1$.

For homogeneous slabs of length $d$, the Beer-Lambert transmittance simplifies to:

$$
T(d) = \exp(-\sigma_t d)
$$

## Angular Phase Functions

### Henyey-Greenstein & Double Henyey-Greenstein

For aerosol droplets and particulate haze characterized by mean scattering cosine $g \in (-1, 1)$:

$$
p_{\text{HG}}(\mu, g) = \frac{1}{4\pi} \frac{1 - g^2}{\left(1 + g^2 - 2g\mu\right)^{3/2}}, \quad \mu = \cos\theta
$$

Notice that in dense fog ($g \approx 0.85 \text{ to } 0.90$), forward scattering dominates by multiple orders of magnitude over side scatter, while weak backward glory peaks occur near $\mu = -1$. Shroud implements `double_henyey_greenstein` to linearly blend a sharp forward diffraction lobe $g_f > 0$ with a backscatter lobe $g_b < 0$:

$$
p_{\text{DHG}}(\mu) = w_f \, p_{\text{HG}}(\mu, g_f) + (1 - w_f) \, p_{\text{HG}}(\mu, g_b)
$$

### Molecular Rayleigh Scattering

For atmospheric gas molecules far smaller than the optical wavelength ($\alpha = 2\pi r / \lambda \ll 1$), dipolar scattering follows:

$$
p_{\text{Rayleigh}}(\mu) = \frac{3}{16\pi} \left(1 + \mu^2\right)
$$

implemented across both `shroud-core::atmospheric::rayleigh_phase` and `shaders/volumetric/phase_rayleigh.wgsl`.

## Null-Collision Unbiased Volume Tracking

For spatially varying volumetric fields $\sigma_t(\mathbf{x})$ where analytical line integration is intractable:

1. **Woodcock Delta Tracking (`shaders/volumetric/delta_tracking_woodcock.wgsl`)**: Pads the medium with fictitious null particles $\sigma_n(\mathbf{x}) = \sigma_{\text{maj}} - \sigma_t(\mathbf{x}) \ge 0$ up to a majorant bound $\sigma_{\text{maj}}$. Free-flight steps are sampled as $\Delta s = -\ln(1 - \xi) / \sigma_{\text{maj}}$ and accepted as real interactions with probability $\sigma_t(\mathbf{x}) / \sigma_{\text{maj}}$.
2. **Ratio Tracking (`shaders/volumetric/ratio_tracking_transmittance.wgsl`)**: Replaces binary rejection along shadow rays with a continuous product estimator over null events:

$$
\hat{T} = \prod_{k} \left(1 - \frac{\sigma_t(\mathbf{x}_k)}{\sigma_{\text{maj}}}\right)
$$

yielding significantly lower variance for LiDAR attenuation and shadow visibility checks.
3. **Spectral Hero Wavelength Tracking (`shaders/volumetric/spectral_hero_tracking.wgsl`)**: Evaluates correlated multi-band extinction along a shared null-collision random walk guided by a sampled hero wavelength, avoiding chromatic noise in dispersive fog and Rayleigh twilight skies.

## Atmospheric Density Stratification

Altitude-dependent molecular density $\rho(h) / \rho_0$ uses the US Standard Atmosphere 1976 model (`crates/shroud-core/src/atmospheric.rs`):
- **Troposphere ($h \le 11\text{ km}$)**: Constant lapse rate $L = 6.5\text{ K/km}$ from sea-level temperature $T_0 = 288.15\text{ K}$.
- **Lower Stratosphere ($h > 11\text{ km}$)**: Exponential isothermal scale-height decay anchored at $H_s = 6340\text{ m}$.
