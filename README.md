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
