# WMP M11 Reward/AMP Alignment State

Date: 2026-06-25

## Completed

- Aligned WMP task reward structure toward the original WMP/A1AMPCfg terms.
- Added optional terrain metadata plumbing through TerrainOutput, TerrainGenerator,
  and TerrainEntity.
- Added WMP gap/pit x-edge masks and replaced the placeholder feet_edge reward with
  contact + terrain type + terrain level + edge-mask logic.
- Added torques, feet_stumble, cheat, stuck, WMP feet_edge reward curriculum, and a
  WMP-local only-positive reward clipping correction term.
- Added AMP expert motion joint position diagnostics and optional 12-dof
  expert_joint_pos_scale / expert_joint_pos_bias configuration.

## Verification

- `uv --cache-dir .uv-cache run python -m py_compile ...`: passed.
- `uv --cache-dir .uv-cache run ruff check ...`: passed.
- `uv --cache-dir .uv-cache run pytest tests\test_wmp_task.py tests\test_terrain_proportion_spawning.py tests\test_rewards.py -q`: 42 passed.
- 2-env CPU real WMP env reset/step smoke: passed; reward finite, terrain metadata
  contains `wmp_edge_resolution` and `wmp_x_edge_mask`.

## Notes

- AMP retargeting remains disabled by default; diagnostics now expose whether
  motion joint distributions differ substantially from the Go1 default posture.
- This slice does not change AMP discriminator reward formula or world-model losses.
