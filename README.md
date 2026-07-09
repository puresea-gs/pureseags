<h1 align="center">PureSeaGS</h1>
<h3 align="center">Decoupled Physics-Aware 3D Gaussian Splatting<br>for Underwater Scene Reconstruction</h3>

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><strong>📄 Paper</strong></a> |
  <a href="https://puresea-gs.github.io/pureseags/"><strong>🌐 Project Page</strong></a> |
  <a href="https://github.com/puresea-gs/pureseags"><strong>💻 Code</strong></a>
</p>

<br>

<p align="center">
  <img src=".assets/pipeline.png" width="100%" />
</p>

<p align="center">
  PureSeaGS reconstructs clean 3D underwater scenes by decoupling the continuous water medium from discrete scene geometry.
</p>

---

## 📌 Release Plan

- [ ] Training code
- [ ] Evaluation scripts

---

## 🗂️ Dataset

We use the [SeaThru-NeRF](https://sea-thru-nerf.github.io/) dataset. Preprocessed scenes follow the COLMAP-style layout:

```
scene/
  images/
    000001.png
    000002.png
    ...
  sparse/
    0/
      cameras.bin
      images.bin
      points3D.bin
```

---

## 🛠️ Installation

PureSeaGS is built on [Nerfstudio](https://docs.nerf.studio/) and [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

```bash
conda create --name pureseags -y python=3.8
conda activate pureseags

# PyTorch + CUDA
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# tiny-cuda-nn + nerfstudio
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install nerfstudio==1.1.4
ns-install-cli

# PureSeaGS
git clone https://github.com/puresea-gs/pureseags.git
cd pureseags
pip install -e .
```

---

## 🚀 Quick Start

```bash
ns-train pureseags --vis viewer colmap \
  --data /path/to/scene \
  --images-path images
```

---

## 📚 Citation

```bibtex
@article{pureseags2025,
  title   = {PureSeaGS: Decoupled Physics-Aware 3D Gaussian Splatting
             for Underwater Scene Reconstruction},
  author  = {LZF},
  journal = {Under Review},
  year    = {2025}
}
```

---

## 🙏 Acknowledgements

This project builds upon [WaterSplatting](https://github.com/water-splatting/water-splatting), [Nerfstudio](https://docs.nerf.studio/), and [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

---

## 📄 License

Apache License 2.0. See [LICENSE](LICENSE).
