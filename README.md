# 3D Gaussian Splatting From Scratch

A PyTorch implementation of 3D Gaussian Splatting built on top of a from-scratch
Structure-from-Motion pipeline. Custom differentiable rasterizer, custom alpha
compositing — no gsplat, no diff-gaussian-rasterization, no nerfstudio.

Follow-up to [sfm-from-scratch](https://github.com/aaryadevchandra/sfm-from-scratch).

![3dgs](readme_files/3dgs.gif)

---

## About The Project

Given a sequence of images from the **DTU dataset**, this pipeline reconstructs a
sparse 3D point cloud using Structure from Motion, places 3D Gaussians on it, and
optimizes those Gaussians against the input views with a hand-written
differentiable rasterizer.

AI note - ONLY the display/plotting functions are ai, rest is all me, feel free to
grill me on each line of the code : )

The pipeline includes:

- SIFT feature detection and mutual descriptor matching
- normalized 8-point Fundamental matrix estimation with RANSAC
- Essential matrix decomposition and relative pose recovery
- custom triangulation from perspective projection equations
- incremental camera registration with RANSAC-based PnP
- cheirality and reprojection filtering on newly triangulated points
- track-based correspondence chaining across frames
- 3D Gaussian initialization from the sparse cloud
- perspective projection of Gaussians into screen-space ellipses
- per-splat bounding boxes with Gaussian falloff evaluation
- depth-sorted front-to-back alpha compositing with transmittance
- Adam optimization of position, scale, opacity and colour
- opacity-based pruning
- `.ply` export for interactive viewing

This is a learning-focused implementation rather than a production renderer.

---

## Dataset

Uses images from the [DTU dataset](https://roboimagedata.compute.dtu.dk/)
(scan114). The dataset is not included in this repo — download it and place it
under `DTU/scan114/image/`.

---

## Results

### Interactive splat viewer

The trained Gaussians exported to `.ply` and viewed in a WebGL splat viewer,
from angles never seen during training:

![3dgs](readme_files/3dgs.gif)


---

## Method Overview

**Structure from Motion.** A seed pair of images gives the initial camera
relationship via the Fundamental and Essential matrices, and matched features are
triangulated into the first sparse cloud. Each subsequent frame is registered by
chaining feature tracks forward — a 3D point's keypoint in frame *n-1* is looked
up in the *n-1 ↔ n* matches to find its pixel in frame *n* — and solving PnP
against those correspondences. Unmatched features are triangulated into new
structure, so the cloud grows as the camera moves.

**Gaussian Splatting.** Each 3D Gaussian is transformed into camera space and
projected to screen space. The projected extent gives an axis-aligned bounding
box, and within it each pixel receives a Gaussian weight from its normalized
distance to the splat centre. Splats are depth-sorted and composited front to
back:

    alpha = opacity * G(x, y)
    C    += T * alpha * colour
    T    *= (1 - alpha)

Transmittance `T` tracks how much light survives to each splat, so occlusion
falls out of the blend rather than needing a separate visibility test. The whole
path is differentiable, so the loss against the ground truth image backpropagates
to every Gaussian's position, scale, opacity and colour.

---

## Known Limitations

Being upfront about what this doesn't do:

- **the rasterizer only uses two of the three scale axes.** `scale[2]` and the
  rotation quaternion never receive gradients, so splats are axis-aligned
  ellipses rather than oriented 3D ellipsoids. This is why the viewer shows
  streaking from angles off the training arc — the third axis is whatever it was
  initialized to. The proper fix is full covariance projection
  (Σ₂D = J·W·Σ·Wᵀ·Jᵀ).
- **no densification.** Reference 3DGS clones Gaussians where the positional
  gradient is large, which is how it resolves fine detail. Without it, detail is
  capped by the initial point cloud.
- **narrow triangulation baseline.** New points are triangulated from adjacent
  frames, which on the DTU rig are only a couple of degrees apart. Depth is
  poorly conditioned as a result and the reconstruction is shallower than the
  true object.
- **pure-Python rasterization loop.** The projection is batched but the
  compositing still loops per splat, so training is slow. Real implementations
  tile the screen and run it in CUDA.
- **no bundle adjustment**, so pose error accumulates along the camera chain.

---

## Built With

- Python
- PyTorch
- NumPy
- OpenCV (SIFT and image IO only)
- SciPy (SLSQP for PnP)
- Plotly

---

## Project Structure

```
.
├── DTU/
│   └── scan114/
│       └── image/
│           ├── 000033.png
│           ├── 000034.png
│           └── ...
├── readme_files/
│   ├── 3dgs.gif
│   ├── point_cloud.gif
│   ├── render_vs_gt.png
│   └── 2d_test.png
├── 3dgs_sfm.ipynb
├── LICENSE
└── README.md
```

---

## Key Learning Goals

Built to understand differentiable rendering from the ground up:

- how a 3D Gaussian becomes a 2D screen-space ellipse under perspective
- why alpha compositing has to be depth-ordered, and what breaks when it isn't
- what transmittance physically represents and why it's multiplicative
- how gradients reach a Gaussian's *position* through a rasterizer
- why unconstrained opacity permanently kills splats under `clamp(min=0)`
- how wrong camera poses show up as view-independent blur rather than as error
- why a shallow point cloud is a symptom of a narrow triangulation baseline

---

## Notes

Most of the geometry and all of the rendering is implemented manually. OpenCV is
used for SIFT extraction and image IO, SciPy for the SLSQP solver inside PnP, and
PyTorch for autodiff and Adam — the rasterizer itself is written from scratch.

Reconstruction quality depends heavily on feature matching, camera baseline, PnP
stability, and the filtering of noisy triangulated points.