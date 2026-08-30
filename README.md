# Mario Kart DS RL
A reinforcement learning agent that learns to drive in Mario using PPO.

## Training

You'll need the game ROM at `files/rom.nds` and at least one savestate (with extension `.noo`) in `savestates/`. The environment picks a random savestate on each reset.

Install dependencies and start training:

```bash
uv sync
uv run python -m src.train
```

Training runs 4 environments in parallel for 20M timesteps, checkpointing to `models/` every 200k steps and recording a video of each new best race under `videos/`. Progress (win rate, average distance, finish times) is logged to TensorBoard:

```bash
uv run tensorboard --logdir mariokartds_tensorboard/
```

To watch a trained model drive:

```bash
uv run src/demo path/tp/checkpoint
```

## Environment

The environment lives in `src/mkds.py` as `MarioKartDSEnv` and is registered as
`MarioKartDS-v0` with a 10000-step episode cap. On reset it loads a random
savestate from the list it was given.

### Observation space

`Box(0, 255, shape=(192, 256, 4), dtype=uint8)` — the raw top screen.
Only the top screen is used; the bottom screen (map) is dropped. During training
this gets run through the usual Atari preprocessing (`WarpFrame` to grayscale
84x84, `MaxAndSkipEnv`, and 4-frame stacking).

### Action space

`Discrete(7)`. The accelerate button (A) is held in every action, so the agent
never has to learn to keep moving; the choices are what to do on top of that:

| Action | Value | Buttons |
| --- | --- | --- |
| NONE | 0 | A |
| LEFT | 1 | A + left |
| RIGHT | 2 | A + right |
| ITEM | 3 | A + X |
| DRIFT | 4 | A + R |
| DRIFT_LEFT | 5 | A + R + left |
| DRIFT_RIGHT | 6 | A + R + right |

### Reward

Distance, speed and lap timing are read straight from game RAM. The reward is a
weighted sum of three components, controlled by `rewards_weight =
(dist, speed, checkpoint)`:

- **distance** — change in track distance since the last frame, clipped to
  [-10, 10] and divided by 100.
- **speed** — distance moved times the current speed, normalised by
  `2**16 * 50`. Rewards actually being fast, not just inching forward.
- **checkpoint** — every 100 units of distance counts as a checkpoint; clearing
  one pays `50 / (steps_in_checkpoint + 5)`, so reaching it in fewer frames is
  worth more.

When `terminate_on_stall` is off, stalling past the timeout applies a small
`-0.01` penalty per frame instead of ending the episode.

### Termination

An episode ends when either:

- the agent finishes the race (crosses the line on lap 3), or
- it stalls — no new max distance for 60 frames — and `terminate_on_stall` is
  on (the default).

Reaching the 10000-step cap truncates the episode.

### Info

Each step returns an info dict with `distance`, `speed`, `total_time_elapsed`,
`lap_time_elapsed`, `lap`, and `win`.
