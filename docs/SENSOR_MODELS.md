# Physical Sensor & Optics Models

Shroud simulates active and passive multi-spectral sensors using first-principles radiometry, coherent signal processing, and detector statistics rather than geometric approximations.

## 1. Pulsed Time-of-Flight (ToF) LiDAR

For a pulsed laser emitting energy $E_p$ at wavelength $\lambda$, the transmitted photon count is $N_{\text{tx}} = E_p \lambda / (h c)$. For an extended diffuse Lambertian target at range $R$ with hemispherical reflectance $\rho$, surface incidence cosine $\cos\theta_i$, receiver aperture area $A_r = \pi D^2 / 4$, optical efficiency $\eta_{\text{opt}}$, and one-way atmospheric transmittance $\tau(R)$, the expected returned photon count is:

$$
N_{\text{rx}} = N_{\text{tx}} \, \rho \cos\theta_i \, \eta_{\text{opt}} \, \tau(R)^2 \, \frac{A_r}{\pi R^2}
$$

Key implementation modules:
- Host equation: `LidarTransceiverConfig::expected_return_photons` in `crates/shroud-core/src/sensors.rs`
- GPU compute kernels: `shaders/sensors/lidar_tof_pulsed.wgsl`, `shaders/sensors/lidar_multi_beam_spin.wgsl`, `shaders/lidar_dsp/peak_detection_cfd.wgsl`
- Full-waveform decomposition: `shaders/lidar_waveforms/full_waveform_gaussian_decomposition.wgsl`

## 2. Coherent FMCW LiDAR & 77 GHz Millimeter-Wave Radar

Frequency-Modulated Continuous-Wave (FMCW) systems transmit triangular linear chirps of bandwidth $B$ over sweep period $T_c$ at carrier $f_0 = c / \lambda$. A target at range $R$ moving at radial velocity $v_r$ produces a range beat frequency and Doppler shift:

$$
f_R = \frac{2 B R}{c T_c}, \qquad f_D = \frac{2 v_r}{\lambda}
$$

Up-chirp and down-chirp intermediate beat frequencies preserve complex I/Q sideband sign:

$$
f_{\text{up}} = f_R - f_D, \qquad f_{\text{down}} = f_R + f_D
$$

Inverting the linear system decouples range and velocity simultaneously:

$$
R = \frac{c T_c}{2 B} \cdot \frac{f_{\text{down}} + f_{\text{up}}}{2}, \qquad v_r = \frac{\lambda}{2} \cdot \frac{f_{\text{down}} - f_{\text{up}}}{2}
$$

**Optical 1550 nm Doppler Note**: At $f_0 = 193.4\text{ THz}$ ($\lambda = 1550\text{ nm}$), even moderate velocities ($12.5\text{ m/s}$) yield $f_D \approx 16.1\text{ MHz}$, which exceeds short-range $f_R$ ($4.0\text{ MHz}$ at $25\text{ m}$). Preserving signed I/Q frequencies in `FmcwConfig::calculate_beat_frequencies` avoids range-Doppler folding.

## 3. Single-Photon Avalanche Diodes (SPAD) & SiPM

In Geiger-mode photon counting (`shaders/sensors/lidar_flash_spad.wgsl`, `shaders/quantum_photodetectors/spad_avalanche_breakdown.wgsl`), avalanche trigger probability follows Poisson statistics over $N_{\text{carriers}}$ primary photo-electrons, followed by hold-off dead-time pileup (`shaders/lidar_waveforms/dead_time_pileup_correction.wgsl`) and stochastic afterpulsing.

## 4. Long-Wave Infrared (LWIR, 8 to 14 um) Thermal Microbolometer

Surface thermal emission integrates Planck's spectral blackbody law across the atmospheric LWIR transmission window $[8\,\mu\text{m}, 14\,\mu\text{m}]$:

$$
L_{\text{LWIR}}(T) = \int_{8\,\mu\text{m}}^{14\,\mu\text{m}} \frac{2 h c^2}{\lambda^5 \left[\exp\left(\frac{h c}{\lambda k_B T}\right) - 1\right]} \, d\lambda
$$

At $T = 300\text{ K}$, numerical midpoint quadrature produces $54.93\text{ W}/(\text{m}^2\cdot\text{sr})$. Apparent sensor radiance accounts for surface emissivity $\varepsilon(\lambda)$ and reflected sky downwelling radiance:

$$
L_{\text{sensor}} = \varepsilon L_{\text{LWIR}}(T_{\text{surface}}) + (1 - \varepsilon) L_{\text{downwelling}}
$$

## 5. Brown-Conrady Camera Lens Distortion

Normalized sensor-plane coordinates $(x, y)$ undergo radial ($k_1, k_2, k_3$) and decentering tangential ($p_1, p_2$) distortion via `DistortionCoefficients::apply`:

$$
x_d = x (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + 2 p_1 x y + p_2 (r^2 + 2 x^2)
$$

$$
y_d = y (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + p_1 (r^2 + 2 y^2) + 2 p_2 x y
$$
