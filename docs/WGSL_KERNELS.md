# WGSL Compute Kernel Corpus Catalog

All **391 standalone WGSL compute modules** (30,513 lines) under `shaders/` parse cleanly and pass full type-checking through `naga 24.0` (`shroud validate shaders/`).

## Subsystem Index

| Directory | Modules | Physical & Algorithmic Scope |
| --- | --- | --- |
| `shaders/common/` | 5 | CODATA constants, aligned GPU structs, Duff ONB, PCG32 RNG, ACES colorimetry |
| `shaders/bvh/` | 10 | 30-bit Morton LBVH, GPU radix sort, Karras tree builder, SAH optimizer, traversal |
| `shaders/wide_bvh/` | 2 | 4-wide (`BVH4`) and 8-wide (`BVH8`) SIMD-style ray-box culling |
| `shaders/tlas/` | 1 | Two-level acceleration structure (TLAS) rigid transform instancing |
| `shaders/materials/` | 10 | Cook-Torrance GGX, Disney Principled, dielectric, conductor, BSSRDF, thin film |
| `shaders/materials_library/` | 40 | Physical optical & radiometric parameters for 40 roadway/automotive/natural materials |
| `shaders/spectral_materials/` | 40 | 3-term Sellmeier dispersion & multi-spectral radiometry across all 40 materials |
| `shaders/volumetric/` | 10 | Henyey-Greenstein, Rayleigh, Mie, Woodcock tracking, ratio tracking, hero spectral |
| `shaders/aerosol_models/` | 10 | Urban smog, maritime spray, desert dust, wildfire smoke, volcanic ash, sea fog |
| `shaders/atmospheric_microphysics/` | 5 | Koehler hygroscopic swelling, lognormal aerosols, Marshall-Palmer rain attenuation |
| `shaders/atmospheric_radiative_equilibrium/` | 7 | HITRAN Voigt profiles, delta-Eddington two-stream, correlated-k, Angstrom AOD |
| `shaders/restir/` | 8 | Spatiotemporal reservoir resampling (ReSTIR DI & ReSTIR GI) with visibility checks |
| `shaders/radiance_cache/` | 5 | World-space irradiance probes & Spherical Gaussian lobe encoding |
| `shaders/spherical_harmonics/` | 2 | L2 spherical harmonics irradiance evaluation & Wigner rotation |
| `shaders/denoise/` | 5 | SVGF temporal accumulation, spatial variance, a-trous wavelet, bilateral, TAA |
| `shaders/postprocess/` | 1 | Dual-Kawase optical bloom & lens halation filter |
| `shaders/sensors/` | 9 | Pulsed ToF LiDAR, heterodyne FMCW, flash SPAD, LWIR microbolometer, Brown-Conrady |
| `shaders/sensor_pipelines/` | 15 | Production models: Hesai Pandar128, Ouster OS1-64, Luminar Iris, Aeva Aeries II, FLIR Boson |
| `shaders/sensor_calibration/` | 10 | Flat-field vignetting, FPN NUC, ToF intensity walk, 3x3 CCM, stereo rectification |
| `shaders/lidar_dsp/` | 1 | Constant-fraction discriminator (CFD) sub-bin echo peak detection |
| `shaders/lidar_waveforms/` | 10 | Multi-Gaussian Levenberg-Marquardt waveform decomposition, Wiener IRF deconvolution |
| `shaders/lidar_slam/` | 6 | LOAM curvature extraction, SLERP motion deskewing, Scan Context polar loop closure |
| `shaders/fmcw_dsp/` | 1 | Windowed Doppler FFT & chirp beat frequency demodulation |
| `shaders/automotive_radars/` | 5 | 4D MIMO virtual array synthesis, CA/OS-CFAR detector, Boulic micro-Doppler, PO RCS |
| `shaders/sar_interferometry/` | 10 | Range-Doppler compression, InSAR coherence, Goldstein phase filter, Cloude-Pottier, BP |
| `shaders/sensor_fusion/` | 10 | 15-state IMU/LiDAR EKF, point-to-plane ICP, 3D NDT, RANSAC ground plane, CI fusion |
| `shaders/compute/` | 6 | Workgroup Blelloch prefix scan, parallel reduction, spatial hash, tiled GEMM, FFT |
| `shaders/numerical_solvers/` | 5 | Preconditioned Conjugate Gradient (PCG), CSR SpMV, Jacobi, Thomas tridiagonal, RK4 |
| `shaders/monte_carlo_samplers/` | 10 | Equiangular light sampling, BDPT vertex, Shirley-Chiu concentric disk, Veach MIS |
| `shaders/qmc_sobol/` | 1 | 32-dimensional Sobol low-discrepancy quasi-Monte Carlo direction vectors |
| `shaders/vndf_sampling/` | 1 | Heitz GGX visible normal distribution function (VNDF) importance sampling |
| `shaders/optical_wave_propagation/` | 5 | Scalar Fresnel diffraction, angular spectrum propagation, Beer-Lambert optical thickness |
| `shaders/wave_optics/` | 6 | Zernike aberrations, Chirp Z-Transform Fresnel, Fraunhofer spikes, Stokes-Mueller, TMM |
| `shaders/spectral_optics/` | 5 | Chromatic aberration, Kogelnik RCWA volume gratings, QWOT AR stacks, Drude-Lorentz |
| `shaders/nonlinear_optics/` | 6 | Quasi-phase-matched SHG, Kerr B-integral self-focusing, degenerate FWM, SBS, soliton |
| `shaders/photonic_integrated_circuits/` | 6 | SOI rib waveguide modes, microring resonators, Clements MZI meshes, AWG, OPA LiDAR |
| `shaders/adaptive_optics/` | 6 | Shack-Hartmann centroids, von Karman phase screens, deformable mirror, sodium LGS |
| `shaders/computational_microscopy/` | 8 | Fourier ptychography, SIM super-resolution, SD-OCT k-space, SPIM, DIC, TIRF, STED, DHM |
| `shaders/hyperspectral/` | 10 | FCLS unmixing, Spectral Angle Mapper, TES LWIR emissivity, APDA vapor, RX anomaly |
| `shaders/quantum_photodetectors/` | 4 | Geiger SPAD quenching, SiPM optical crosstalk, EMCCD Poisson-Gamma, streak camera |
| `shaders/quantum_optics/` | 10 | Type-I/II SPDC, Hong-Ou-Mandel dip, MLE tomography, Wigner phase space, CHSH Bell |
| `shaders/cryogenic_superconducting/` | 7 | SNSPD NbN hotspot, TES bolometer ETF, Josephson RCSJ, London screening, DC SQUID |
| `shaders/acoustic_sonar/` | 10 | Bellhop SSP ray tracing, Thorp oceanic attenuation, DAS beamforming, SAS backprojection |
| `shaders/oceanic_hydrology/` | 4 | Petzold ocean VSF, water vibrational Raman shift, Pope-Fry chlorophyll/CDOM, Cox-Munk |
| `shaders/astronomy_sky/` | 4 | Bruneton precomputed atmospheric transmittance & inscatter, zodiacal dust, airglow |
| `shaders/stellar_astrophysics/` | 8 | Kurucz ATLAS9 opacity, CME MHD shock, Mandel-Agol transit, Shakura-Sunyaev, CCM dust |
| `shaders/optical_communication/` | 7 | FSO link budget, 16-QAM coherent equalizer, NLSE fiber propagation, EDFA ASE, Gamma-Gamma |
| `shaders/geophysics_seismology/` | 10 | Navier-Cauchy P/SV split, Rayleigh dispersion, TTI wavefield, FWI adjoint, FMM eikonal |
| `shaders/plasma_aerodynamics/` | 8 | Re-entry plasma blackout, Saha ionization, Hartmann MHD flow, Debye sheath, Langmuir |
| `shaders/metrology_sensors/` | 5 | Heterodyne phase unwrapping, chromatic confocal, 5-step Hariharan Fizeau, VSI, BA |
| `shaders/thermal_physics/` | 1 | Steady-state conductive heat diffusion equilibrium solver |
