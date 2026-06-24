# WMP AMP motion data

Place WMP-compatible AMP motion `.txt` files here, or set `WMP_MOTION_DATA_DIR`
to a directory containing the original `datasets/mocap_motions/*.txt` files.

The runner accepts either already-extracted AMP observation transitions
`[amp_obs_t, amp_obs_t+1]` or frame-wise AMP observations with at least the
configured AMP observation dimension.
