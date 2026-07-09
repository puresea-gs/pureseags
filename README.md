<h1 align="center">PureSeaGS</h1>
<h3 align="center">Decoupled Physics-Aware 3D Gaussian Splatting<br>for Underwater Scene Reconstruction</h3>

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><strong>📄 Paper</strong></a> |
  <a href="https://puresea-gs.github.io/pureseags/"><strong>🌐 Project Page</strong></a>
</p>

<br>

<p align="center">
  <img src=".assets/pipeline.png" width="100%" />
</p>

---

## Abstract

Underwater 3D reconstruction inherently suffers from the physical conflict between continuous optical media and existing discrete representation methods. Applying explicit 3D Gaussian Splatting (3DGS) to continuous water volumes inevitably introduces a **representation mismatch**. This compromise causes severe **gradient entanglement** during optimization, forcing the model to generate semi-transparent floater artifacts to compensate for medium absorption.

To resolve this at the architectural level, we propose **PureSeaGS**, a physics-aware and prior-driven decoupled underwater 3DGS framework. The core of our approach is the **Anisotropic Medium Field (AMF)** with separate prediction heads for distinct physical quantities:

| Physical quantity | Symbol | Description |
|---|---|---|
| Medium color | `C_med` | Base color of the water body (analogous to veiling light) |
| Spectral attenuation | `σ_attn` | Wavelength-dependent decay of object radiance with distance |
| Effective backscattering | `σ_bs` | Accumulation rate of ambient light scattered into the line-of-sight |

Instead of merely modeling scattering directionality, the AMF achieves **feature-level decoupling** of physical parameters, effectively severing gradient interference between the continuous water volume and discrete scene geometry during backpropagation. We further introduce a **depth-guided enhancement mechanism** to condition medium predictions on scene geometry, and a **physics-regularized loss formulation** to stabilize optimization under severe underwater appearance variations.

Extensive experiments demonstrate that PureSeaGS effectively suppresses semi-transparent floater artifacts and preserves the underlying 3D geometric structures. Our method achieves state-of-the-art novel view synthesis performance, reaching a **PSNR of 32.40 dB** on the Curaçao dataset — a significant margin of **0.83 dB** over the best existing baseline — while fully maintaining the real-time rendering efficiency of the native 3DGS rasterizer.

## Method Overview

PureSeaGS follows a **decouple-then-couple** strategy with four stages:

1. **State Encoding** — SH-encoded ray directions capture view-dependent anisotropic scattering characteristics
2. **Physics-Decoupled Medium MLP (AMF)** — Shared backbone routes into three independent prediction heads, eliminating gradient interference between physically distinct quantities
3. **Coupled Volumetric Field Rendering** — Physically integrates object radiance and medium scattering via the Jaffe-McGlamery underwater image formation model
4. **Joint Loss Constraints** — Regularized L1 + SSIM with adaptive structure penalty and depth-guided conditioning

## Key Results

- **Novel View Synthesis:** PSNR 32.40 dB on Curaçao (+0.83 dB over SOTA), real-time rendering maintained
- **Floater Suppression:** Semi-transparent artifacts substantially eliminated via architectural decoupling
- **Depth Preservation:** Clean depth maps with sharp geometric boundaries, unlike prior methods that sacrifice geometry to explain the medium
- **Color Fidelity:** Accurate water-free scene restoration validated on DKC-Pro color chart

## Code

Full source code will be released upon paper acceptance. Helper scripts for data preprocessing and evaluation are provided in this repository.

## Citation

```bibtex
@article{pureseags2025,
  title   = {PureSeaGS: Decoupled Physics-Aware 3D Gaussian Splatting
             for Underwater Scene Reconstruction},
  author  = {LZF},
  journal = {Under Review},
  year    = {2025}
}
```

## Acknowledgements

This work builds upon [WaterSplatting](https://github.com/water-splatting/water-splatting) (3DV 2025), [Nerfstudio](https://docs.nerf.studio/), and [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). We thank the [SeaThru-NeRF](https://sea-thru-nerf.github.io/) authors for open-sourcing their dataset.

## License

Apache License 2.0. See [LICENSE](LICENSE).
