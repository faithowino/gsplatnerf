# Gaussian Splatting Experiment

## Overview

This project investigates how the amount of training imagery affects the quality of 3D scene reconstruction using **3D Gaussian Splatting (3DGS)**.

The central research question is:

> **How many training images does Gaussian Splatting need to produce a high-quality 3D reconstruction, and at what point does reducing the number of images cause significant quality degradation?**

The experiment systematically trains Gaussian Splatting models using different numbers of training images while keeping the training procedure consistent. Multiple random seeds are used to evaluate the reproducibility of the results.

The project evaluates reconstruction quality quantitatively using **PSNR** and **SSIM**, while also generating consistent rendered images for qualitative comparison.

The resulting data can be used to study:

- The relationship between training image count and reconstruction quality.
- The minimum amount of training data required for acceptable reconstruction.
- The point at which reconstruction quality begins to degrade substantially.
- Whether additional training images continue to provide meaningful improvements.
- The consistency of results across different random seeds.
- The effect of training-data reduction on the number of learned Gaussian primitives.

---

## Research Framework

The experiment uses a fixed set of training-image budgets:

```text
5
15
30
45
60
90
120
150
161
```

For the main quantitative experiment, each image count is evaluated using multiple random seeds:

```text
Seed 0
Seed 1
Seed 2
Seed 3
Seed 4
```

This produces:

```text
9 image counts × 5 seeds = 45 experimental runs
```

Each run is trained for a fixed number of optimization steps and evaluated using the same evaluation procedure.

The original Garden experiment was used as the initial test case. The same framework can subsequently be applied to additional scenes such as:

```text
bicycle
bonsai
counter
kitchen
room
stump
```

The scenes contain different numbers of available training images, so the image subsets are defined relative to the number of images available in each scene.

---

## Method

For each experiment, the pipeline:

1. Selects a specified number of training images.
2. Uses a specified random seed.
3. Trains a 3D Gaussian Splatting model.
4. Saves the resulting checkpoint.
5. Evaluates the trained model using held-out validation views.
6. Records quantitative metrics.
7. Generates rendered validation images for qualitative comparison.
8. Stores the results in a structured directory.

The same validation camera views are used across different image-count experiments so that rendered results can be compared directly.

---

## Quantitative Evaluation

The experiment currently uses:

### PSNR

**Peak Signal-to-Noise Ratio (PSNR)** measures the pixel-level similarity between the rendered image and the corresponding ground-truth image.

Higher PSNR indicates better reconstruction quality.

### SSIM

**Structural Similarity Index (SSIM)** measures structural similarity between the rendered and ground-truth images.

Higher SSIM indicates greater similarity.

### Gaussian Count

The number of Gaussian primitives in the trained model is also recorded.

This provides an additional measure of how the representation changes as the amount of training data is reduced.

---

## Qualitative Evaluation

In addition to numerical metrics, the evaluation pipeline can render the validation views from each trained checkpoint.

For each experiment, the rendered images are stored in:

```text
renders/
```

and the corresponding ground-truth images are stored in:

```text
ground_truth/
```

For example:

```text
results/
└── starvation/
    └── garden_keep_60_seed_0/
        ├── ckpts/
        ├── renders/
        │   ├── 000.png
        │   ├── 001.png
        │   └── ...
        ├── ground_truth/
        │   ├── 000.png
        │   ├── 001.png
        │   └── ...
        └── stats/
            └── custom_eval.json
```

Because the rendered and ground-truth images correspond to the same validation cameras, they can be directly compared.

For publication figures, a single seed can be used for qualitative visualization while all seeds are retained for quantitative statistical analysis.

---

## Technologies

This project uses:

- **Python** — experiment automation and evaluation
- **PyTorch** — model execution and GPU computation
- **3D Gaussian Splatting (3DGS)** — scene representation and rendering
- **GSplat** — Gaussian Splatting implementation
- **NumPy** — numerical analysis
- **Pillow** — saving rendered images
- **CSV / JSON** — experiment results and metadata
- **Bash** — experiment and rendering automation
- **CUDA** — GPU acceleration
- **Mip-NeRF 360 dataset** — multi-view scene data
- **LLM-assisted scripting** — used to assist with implementation and automation of the experimental pipeline

---

## Dataset

The experiments use scenes from the **360_v2** portion of the Mip-NeRF 360 dataset.

The local dataset structure is expected to resemble:

```text
examples/data/360_v2/
├── bicycle/
├── bonsai/
├── counter/
├── garden/
├── kitchen/
├── room/
└── stump/
```

Each scene contains the image data, camera information, and sparse reconstruction data required by the training pipeline.

---

## Requirements

A Linux environment with an NVIDIA GPU is recommended.

The project requires:

- Python 3
- PyTorch
- CUDA-compatible NVIDIA GPU
- GSplat / GSplat-NeRF environment
- Mip-NeRF 360 scene data

The exact Python dependencies are determined by the GSplat-NeRF environment used for the experiments.

---

# Installation

## 1. Clone the project

```bash
git clone <repository-url>
cd gsplatnerf
```

If the project already exists locally, navigate to the project directory:

```bash
cd ~/code/gsplatnerf
```

---

## 2. Create/activate the Python environment

The experiments were run inside a Python virtual environment.

For example:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies according to the GSplat-NeRF project configuration.

If the project already contains an environment, activate it instead:

```bash
source .venv/bin/activate
```

---

## 3. Verify the installation

From the project root:

```bash
python examples/simple_trainer.py --help
```

The command should display the available training options.

---

# Running the Experiment

The main experiment is controlled by:

```text
run_starvation_experiment.py
```

The experiment configuration specifies:

- Dataset scene
- GPU
- Training step count
- Image-count subsets
- Random seeds
- Output directory

For example, the original Garden experiment used:

```text
Image counts:
161, 150, 120, 90, 60, 45, 30, 15, 5

Seeds:
0, 1, 2, 3, 4

Maximum training steps:
7000
```

Run the experiment with:

```bash
python run_starvation_experiment.py
```

The script automatically creates a separate result directory for each image-count/seed combination.

---

# Results

Results are stored under:

```text
results/starvation/
```

A typical experiment directory looks like:

```text
results/starvation/
├── garden_keep_5_seed_0/
├── garden_keep_15_seed_0/
├── garden_keep_30_seed_0/
├── garden_keep_45_seed_0/
├── garden_keep_60_seed_0/
├── garden_keep_90_seed_0/
├── garden_keep_120_seed_0/
├── garden_keep_150_seed_0/
└── garden_keep_161_seed_0/
```

Each experiment contains its trained checkpoint and evaluation output.

---

# Evaluating an Existing Checkpoint

The evaluation script can load an existing checkpoint rather than retraining the model.

For example:

```bash
PYTHONPATH=examples \
CUDA_VISIBLE_DEVICES=0 \
python examples/my_eval.py \
    --result_dir results/starvation/garden_keep_60_seed_0
```

The evaluator automatically searches the experiment's `ckpts/` directory for the newest checkpoint.

This makes it possible to regenerate evaluation results or rendered images from previously completed experiments without repeating training.

---

# Rendering Existing Checkpoints

The evaluation script can also generate rendered validation images.

For example:

```bash
PYTHONPATH=examples \
CUDA_VISIBLE_DEVICES=0 \
python examples/my_eval.py \
    --result_dir results/starvation/garden_keep_60_seed_0
```

The rendered images are saved under:

```text
results/starvation/garden_keep_60_seed_0/renders/
```

The corresponding ground-truth images are saved under:

```text
results/starvation/garden_keep_60_seed_0/ground_truth/
```

A Bash script can be used to render the checkpoints for one seed:

```bash
./render_seed0.sh
```

This generates qualitative results for all image-count conditions without retraining the models.

---

# Output Data

The quantitative results are stored in CSV format, with fields including:

```text
experiment
seed
num_images
psnr
ssim
lpips
num_GS
step
checkpoint
```

Example:

```text
garden_keep_60_seed_0,0,60,24.9960,0.78399,,2193390,6999,...
```

The experiment can therefore be analyzed using Python, Jupyter Notebook, pandas, or other statistical tools.

---

# Research Analysis

The primary analysis focuses on plotting reconstruction quality against the number of training images.

For example:

```text
Number of Training Images
          ↓
PSNR / SSIM
          ↓
Reconstruction Quality
```

The five random seeds allow the experiment to report:

- Mean PSNR
- PSNR standard deviation
- Mean SSIM
- SSIM standard deviation

This provides a more robust estimate than relying on a single training run.

The qualitative renders complement these statistics by showing how visual reconstruction quality changes as the training image budget decreases.

---

# Project Goals

The long-term goal of this experiment is to identify three important regions in the image-data/quality relationship:

### 1. Minimum viable data

The smallest number of training images that still produces an acceptable reconstruction.

### 2. Quality degradation region

The range where reducing the number of training images causes substantial reconstruction artifacts or metric degradation.

### 3. Diminishing returns

The point where adding additional training images produces relatively small improvements in reconstruction quality.

Together, these results can provide an empirical characterization of the **data efficiency of 3D Gaussian Splatting for novel view synthesis**.

---

# Project Structure

```text
gsplatnerf/
│
├── examples/
│   ├── simple_trainer.py
│   ├── my_eval.py
│   └── data/
│       └── 360_v2/
│           ├── bicycle/
│           ├── bonsai/
│           ├── counter/
│           ├── garden/
│           ├── kitchen/
│           ├── room/
│           └── stump/
│
├── run_starvation_experiment.py
├── render_seed0.sh
│
└── results/
    └── starvation/
        ├── garden_keep_5_seed_0/
        ├── garden_keep_15_seed_0/
        ├── ...
        └── garden_keep_161_seed_4/
```

---

# Reproducibility

Each experiment records:

- Scene
- Number of training images
- Random seed
- Training step
- Checkpoint path
- Number of Gaussian primitives
- Evaluation metrics

This allows completed experiments to be inspected, evaluated again, or used for further analysis without retraining.

The experimental design is intended to make comparisons between different training-data budgets consistent and reproducible.

---

## Research Question

The project is ultimately designed to answer:

> **How does reducing the number of training images affect the quality and efficiency of 3D Gaussian Splatting, and what is the minimum amount of image data required to maintain high-quality novel view synthesis?**
























# gsplat

[![Core Tests.](https://github.com/nerfstudio-project/gsplat/actions/workflows/core_tests.yml/badge.svg?branch=main)](https://github.com/nerfstudio-project/gsplat/actions/workflows/core_tests.yml)
[![Docs](https://github.com/nerfstudio-project/gsplat/actions/workflows/doc.yml/badge.svg?branch=main)](https://github.com/nerfstudio-project/gsplat/actions/workflows/doc.yml)

[http://www.gsplat.studio/](http://www.gsplat.studio/)

gsplat is an open-source library for CUDA accelerated rasterization of gaussians with python bindings. It is inspired by the SIGGRAPH paper [3D Gaussian Splatting for Real-Time Rendering of Radiance Fields](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), but we’ve made gsplat even faster, more memory efficient, and with a growing list of new features! 

<div align="center">
  <video src="https://github.com/nerfstudio-project/gsplat/assets/10151885/64c2e9ca-a9a6-4c7e-8d6f-47eeacd15159" width="100%" />
</div>

## News

### Unreleased

Changes on `main` since the [v1.5.3](https://github.com/nerfstudio-project/gsplat/releases/tag/v1.5.3) tag (not yet on PyPI).

- [May 2026] **Inference Rendering (HiGS)** -- An experimental inference-only rendering path based on HiGS (Hierarchically Tiled Gaussian Splatting) is now available under the `experimental` package. The inference path uses macro-tile fused rasterization with fp16 scene packing for low-latency rendering of pre-trained Gaussian scenes. For more details, see the [Inference Rendering](#inference-rendering) section and the [HiGS project page](https://research.nvidia.com/labs/sil/projects/higs/).
- [May 2026] Native CUDA **MCMC perturb** (`inject_noise`) speeds up the noise-injection step used in MCMC-style Gaussian optimization.
- [Apr 2026] **AccuTile** adds a conservative ellipse-based tile–Gaussian intersection test on the 3DGS path for tighter work scheduling before rasterization ([PR #927](https://github.com/nerfstudio-project/gsplat/pull/927)).
- [Apr 2026] **NCore v4** capture support, including richer camera models and point-cloud loading via `PointCloudsSourceProtocol`. See the [NCore example](https://docs.gsplat.studio/main/examples/ncore.html).
- [Mar 2026] **LiDAR** rasterization for 3D Gaussian splatting: spinning-lidar camera models, `eval3d` rendering, depth / hit-distance modes, and related tooling (optional SciPy via `pip install "gsplat[lidar]"`).
- [Mar 2026] **TorchScript-oriented** deployment: camera models and distortions are also available through PyTorch **custom operators** and **custom classes**, not only Python callables.
- [Mar 2026] **3DGUT** extensions: [external distortion](https://github.com/nerfstudio-project/gsplat/pull/886) (e.g. windshield-style rigs), optional **per-ray** inputs with gradients, optional **ray-normal** outputs, and refactored render modes / extra signals (see [3DGUT notes](docs/3dgut.md)).
- [Jan 2026] [PPISP](https://research.nvidia.com/labs/sil/projects/ppisp/) is integrated as an alternative way of bilateral grid to compensate the training views.

### v1.5.3

- [May 2025] Arbitrary batching (over multiple scenes and multiple viewpoints) is supported now!! Checkout [here](docs/batch.md) for more details! Kudos to [Junchen Liu](https://junchenliu77.github.io/).
- [May 2025] [Jonathan Stephens](https://x.com/jonstephens85) makes a great [tutorial video](https://www.youtube.com/watch?v=ACPTiP98Pf8) for Windows users on how to install gsplat and get start with 3DGUT.
- [April 2025] [NVIDIA 3DGUT](https://research.nvidia.com/labs/toronto-ai/3DGUT/) is now integrated in gsplat! Checkout [here](docs/3dgut.md) for more details. [[NVIDIA Tech Blog]](https://developer.nvidia.com/blog/revolutionizing-neural-reconstruction-and-rendering-in-gsplat-with-3dgut/) [[NVIDIA Sweepstakes]](https://www.nvidia.com/en-us/research/3dgut-sweepstakes/)

## Installation

**Dependence**: Please install [Pytorch](https://pytorch.org/get-started/locally/) first.

The easiest way is to install from PyPI. In this way it will build the CUDA code **on the first run** (JIT).

```bash
pip install gsplat
```

Alternatively you can install gsplat from source. In this way it will build the CUDA code during installation.

```bash
pip install git+https://github.com/nerfstudio-project/gsplat.git
```

We also provide [pre-compiled wheels](https://docs.gsplat.studio/whl) for both linux and windows on certain python-torch-CUDA combinations (please check first which versions are supported). Note this way you would have to manually install [gsplat's dependencies](https://github.com/nerfstudio-project/gsplat/blob/6022cf45a19ee307803aaf1f19d407befad2a033/setup.py#L115). For example, to install gsplat for pytorch 2.0 and cuda 11.8 you can run
```
pip install ninja numpy jaxtyping rich
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt20cu118
```

To build gsplat from source on Windows, please check [this instruction](docs/INSTALL_WIN.md).

## Evaluation

This repo comes with a standalone script that reproduces the official Gaussian Splatting with exactly the same performance on PSNR, SSIM, LPIPS, and converged number of Gaussians. Powered by gsplat’s efficient CUDA implementation, the training takes up to **4x less GPU memory** with up to **15% less time** to finish than the official implementation. Full report can be found [here](https://docs.gsplat.studio/main/tests/eval.html).

```bash
cd examples
pip install -r requirements.txt
# install the scene/stage helper libraries the example trainers import
python -m pip install -e ../libs/scene -e ../libs/stage
# download mipnerf_360 benchmark data
python datasets/download_dataset.py
# run batch evaluation
bash benchmarks/basic.sh
```

## Examples

We provide a set of examples to get you started! Below you can find the details about
the examples (requires installing some extra dependencies via `pip install -r examples/requirements.txt --no-build-isolation`, plus the scene/stage helper
libraries the trainers import via `python -m pip install -e libs/scene -e libs/stage`)

- [Train a 3D Gaussian splatting model on a COLMAP capture.](https://docs.gsplat.studio/main/examples/colmap.html)
- [Fit a 2D image with 3D Gaussians.](https://docs.gsplat.studio/main/examples/image.html)
- [Render a large scene in real-time.](https://docs.gsplat.studio/main/examples/large_scale.html)
- [Train on an NCore v4 capture.](https://docs.gsplat.studio/main/examples/ncore.html)


## Inference Rendering

gsplat includes an experimental inference-only rendering path based on HiGS (Hierarchically Tiled Gaussian Splatting) in the standalone `experimental` package, designed for low-latency rendering of pre-trained Gaussian scenes where training gradients are not needed. The inference path packs scene data into compact fp16 layouts and uses a macro-tile fused rasterization pipeline for fast single-camera rendering.

```python
from experimental import render_scene, GaussianInferenceScene
```

The `simple_viewer.py` example supports the Inference path via the `--use_gaussian_render_inference_scene` flag. A standalone benchmark comparing Inference rendering against the default `rasterization()` path is available in [examples/benchmarks/gaussian_render_inference_scene/](examples/benchmarks/gaussian_render_inference_scene/); run [`gaussian_render_inference_scene_bench.py`](examples/benchmarks/gaussian_render_inference_scene/gaussian_render_inference_scene_bench.py) from the repo root. For more details, see the [HiGS project page](https://research.nvidia.com/labs/sil/projects/higs/).

## Development and Contribution

This repository was born from the curiosity of people on the Nerfstudio team trying to understand a new rendering technique. We welcome contributions of any kind and are open to feedback, bug-reports, and improvements to help expand the capabilities of this software.

This project is developed by the contributors coming from following institutes (unordered):

- UC Berkeley
- NVIDIA
- ShanghaiTech University
- Amazon
- Meta
- IIIT
- LumaAI
- SpectacularAI
- Aalto University
- CMU

We also have a white paper with about the project with benchmarking and mathematical supplement with conventions and derivations, available [here](https://arxiv.org/abs/2409.06765). If you find this library useful in your projects or papers, please consider citing:

```
@article{ye2025gsplat,
  title={gsplat: An open-source library for Gaussian splatting},
  author={Ye, Vickie and Li, Ruilong and Kerr, Justin and Turkulainen, Matias and Yi, Brent and Pan, Zhuoyang and Seiskari, Otto and Ye, Jianbo and Hu, Jeffrey and Tancik, Matthew and Angjoo Kanazawa},
  journal={Journal of Machine Learning Research},
  volume={26},
  number={34},
  pages={1--17},
  year={2025}
}
```

We welcome contributions of any kind and are open to feedback, bug-reports, and improvements to help expand the capabilities of this software. Please check [docs/DEV.md](docs/DEV.md) for more info about development.
