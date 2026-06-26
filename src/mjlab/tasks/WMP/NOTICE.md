# WMP Reproduction Notice

This task package is a MjLab WMP reproduction scaffold for:

- Paper: "World Model-based Perception for Visual Legged Locomotion"
- arXiv: https://arxiv.org/abs/2409.16784
- Reference implementation inspected locally from `E:\开源项目\WMP-master\WMP-master`

The implementation in this directory is written against MjLab's manager-based
environment, sensor API, terrain API, task registry, and training CLI. It
supports both the existing MjLab Unitree Go1 asset and a migrated Unitree A1
asset for closer alignment with the original WMP implementation.

The A1 asset under `src/mjlab/asset_zoo/robots/unitree_a1` is derived from the
local reference repository path:

- `E:\开源项目\WMP-master\WMP-master\resources\robots\a1`

The MJCF file is newly authored for MjLab/MuJoCo training, while the visual
meshes were converted from the upstream A1 mesh files. The upstream A1 license
notice is preserved at:

- `src/mjlab/asset_zoo/robots/unitree_a1/a1_license.txt`

If upstream Dreamer, AMP, mocap, or utility files are copied verbatim into this
repository later, preserve their original license and copyright notices beside
the copied files.
