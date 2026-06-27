# WMP Original Alignment State

## Scope

- Align all WMP tasks with the original WMP training details where practical.
- A1 tasks are the original-equivalent mainline.
- Go1 tasks remain comparison tasks and may keep explicit expert transforms.

## Current Milestone

- M7 complete: Dreamer, ActorCriticWMP dimensions, AMP/PPO joint optimizer,
  critic privileged observations, and reward weights are implemented and
  validated at unit/config/smoke level. WMP privileged randomization cache now
  reads actual post-DR simulator/entity values instead of re-sampling fake values.
  Play/checkpoint loading now ignores transient Dreamer RSSM hidden state and
  shape-checks runner world-model buffers so models trained with 2048 envs can
  be viewed with a different play env count.

## Completed Gates

- M0: Reconstructed current WMP runner/modules/config and original WMP
  Dreamer/ActorCritic/AMPPPO interfaces.
- M1: Vendored original Dreamer package into
  `src/mjlab/tasks/WMP/rl/dreamer` and added `DreamerWorldModelAdapter`
  plus `DreamerEpisodeDataset`.
- M2: Updated `ActorCriticWMP` to original WMP dimensions:
  history latent 35, WM latent 32, WM deterministic feature input 512,
  history encoder `[256, 128]`, WM encoder `[64, 64]`, actor `[256, 128, 64]`,
  critic `[512, 256, 128]`.
- M3: Merged AMP discriminator optimization into the PPO optimizer/loss path.
  The runner now uses one Adam over actor-critic plus discriminator trunk/head,
  writes policy AMP observations to replay, and adds discriminator loss plus
  gradient penalty inside PPO minibatches.
- M4: Extended critic privileged observations with contact flags, contact
  forces, randomized PD gains, base COM/mass, restitution, friction, base
  velocities, actor terms, and height scan.
- M5: Reward structure and weights are locked to the original A1AMPCfg-equivalent
  set, with only-positive reward correction and feet-edge curriculum preserved.
- M6: Replaced independent privileged DR re-sampling with exact post-DR readback:
  PD gain deltas from `actuator_gainprm/biasprm`, motor-strength scale from
  `actuator_forcerange`, base mass from `body_mass`, COM from `body_ipos`, and
  friction from `geom_friction`.
- M7: Fixed play/checkpoint loading for Dreamer RSSM state. Checkpoints save only
  world-model weights/config, old checkpoints with a saved `state` ignore that
  runtime state on load, and `features()` drops any remaining state whose batch
  size differs from the current observations.

## Changed Files

- `WMP_original_alignment_state.md`
- `src/mjlab/tasks/WMP/rl/dreamer/{__init__.py,NOTICE.md,configs.yaml,models.py,networks.py,tools.py}`
- `src/mjlab/tasks/WMP/rl/modules.py`
- `src/mjlab/tasks/WMP/rl/config.py`
- `src/mjlab/tasks/WMP/rl/runner.py`
- `src/mjlab/tasks/WMP/rl/amp.py`
- `src/mjlab/tasks/WMP/mdp/{__init__.py,events.py,observations.py}`
- `src/mjlab/tasks/WMP/wmp_env_cfg.py`
- `tests/test_wmp_task.py`

## Validation Commands

- Passed: `uv --cache-dir .tmp\uv-cache run python -m py_compile ...`
  for touched WMP runner, Dreamer, AMP, MDP, env cfg, and tests.
- Passed: `uv --cache-dir .tmp\uv-cache run pytest --basetemp .tmp\pytest tests\test_a1_constants.py tests\test_asset_zoo.py tests\test_wmp_task.py -q`
  with `27 passed, 6 warnings`.
- Passed: `uv --cache-dir .tmp\uv-cache run pytest --basetemp .tmp\pytest tests\test_wmp_task.py -q`
  with `19 passed, 6 warnings`, including a regression that loads a checkpoint
  carrying stale batch-2048 Dreamer state and runs batch-2 inference.
- Passed: task config load for:
  - `Mjlab-WMP-Rough-Unitree-Go1`
  - `Mjlab-WMP-Stairs-Only-Unitree-Go1`
  - `Mjlab-WMP-Rough-Unitree-A1`
  - `Mjlab-WMP-Stairs-Only-Unitree-A1`
- Passed: A1 stairs AMP real env smoke:
  `uv --cache-dir .tmp\uv-cache run train Mjlab-WMP-Stairs-Only-Unitree-A1 --gpu-ids "[]" --env.scene.num-envs 2 --agent.max-iterations 1 --agent.num-steps-per-env 2 --agent.world-model.train-start-steps 999999 --agent.depth-predictor.camera-num-envs 1 --agent.logger tensorboard --log-root .tmp\wmp_smoke_logs`
  using `WARP_CACHE_PATH=.tmp\warp-cache`.

## Known Limitations

- Dreamer training did not trigger in the tiny env smoke because the original
  episode-level dataset flow inserts sequences on episode reset; the unit test
  validates `DreamerWorldModelAdapter.train_world_model()` directly.
- Restitution is currently cached as the actual configured default `0.0`, because
  WMP has no physical restitution randomization event in the current MjLab DR
  utility set.
- `dr.body_mass` warns that inertia is not scaled; this matches an added-mass
  style perturbation, not a full density perturbation.
- Go1 stairs-only remains the no-AMP diagnostic task. A1 rough and A1 stairs
  enable AMP with reward scale `0.01`.

## Next Action

- For a stricter long-training parity pass, add a real restitution DR mutator
  if restitution should be randomized physically, and add a forced-reset Dreamer
  rollout smoke that exercises episode insertion plus world-model training inside
  `WMPRunner`.
