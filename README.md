# QFlow

QFlow is a flow-matching offline RL algorithm that uses Q-value gradients to guide the flow actor toward high-value actions during inference.

## Installation

This repository is built on the [FQL framework](https://github.com/seohongpark/fql), requiring Python 3.9+ and JAX. The main dependencies are
`jax >= 0.4.26`, `ogbench == 1.1.0`, and `gymnasium == 0.29.1`.

```bash
pip install -r requirements.txt
```

> **Note:** To use D4RL environments, you need to additionally set up MuJoCo 2.1.0.

## Usage

The main implementation is in [agents/qflow.py](agents/qflow.py) with baselines (FQL, IFQL, FBRAC, FAWAC, QIPO-OT, ReBRAC, IQL).

### QFlow hyperparameter config

Per-domain hyperparameters for QFlow are managed in `qflow_config.py`.
When `--agent=agents/qflow.py` is passed, `main.py` automatically loads the matching
entry from `QFLOW_PARAMS` based on the environment name. Please refer to the paper (Q-Flow: Stable and Expressive Reinforcement Learning with Flow-based Policy) for specific hyperparameters used in main experiments.

To override any parameter from the command line, simply pass it as a flag — it takes
precedence over the config file:

```bash
# Uses qflow_config.py values for antmaze-large-play
python main.py --env_name=antmaze-large-play-v2 --agent=agents/qflow.py

# Override alpha and q_agg from the command line
python main.py --env_name=antmaze-large-play-v2 --agent=agents/qflow.py \
    --agent.alpha=0.5 --agent.q_agg=min
```

To add a new domain, add an entry to `QFLOW_PARAMS` in `qflow_config.py`:

```python
"my-new-env": dict(
    alpha=0.2,
    use_time_embed=True,
    time_embed_dim=16,
    q_agg='mean',
    actor_loss_type="grad",
    discount=0.99,
),
```

### Offline RL

```bash
# QFlow on OGBench antmaze-large-navigate-singletask-task1-v0
python main.py --env_name=antmaze-large-navigate-singletask-task1-v0 --agent=agents/qflow.py

# QFlow on OGBench cube-double-play-singletask-task1-v0
python main.py --env_name=cube-double-play-singletask-task1-v0 --agent=agents/qflow.py

# QFlow on OGBench puzzle-4x4-play-singletask-task1-v0
python main.py --env_name=puzzle-4x4-play-singletask-task1-v0 --agent=agents/qflow.py

# QFlow on D4RL antmaze-umaze-v2
python main.py --env_name=antmaze-umaze-v2 --agent=agents/qflow.py

# QFlow on D4RL antmaze-large-diverse-v2
python main.py --env_name=antmaze-large-diverse-v2 --agent=agents/qflow.py
```

### Offline-to-online RL

Pass `--online_steps` to continue training online after the offline phase:

```bash
# QFlow offline-to-online on OGBench cube-double-play-singletask-task1-v0
python main.py --env_name=cube-double-play-singletask-task1-v0 --agent=agents/qflow.py \
    --offline_steps=1000000 --online_steps=1000000

# QFlow offline-to-online on OGBench scene-play-singletask-task1-v0
python main.py --env_name=scene-play-singletask-task1-v0 --agent=agents/qflow.py \
    --offline_steps=1000000 --online_steps=1000000

# QFlow offline-to-online on D4RL antmaze-medium-diverse-v2
python main.py --env_name=antmaze-medium-diverse-v2 --agent=agents/qflow.py \
    --offline_steps=1000000 --online_steps=200000
```

### Running with scripts

Bash scripts for all agents are provided under `scripts/`:

```bash
# Run QFlow (uses qflow_config.py for hyperparameters)
bash scripts/run_qflow.sh 1 5        # tasks 1–5
bash scripts/run_qflow.sh 1 1 debug  # debug mode (seed 0 only)

# Run baselines
bash scripts/run_fql.sh 1 5
bash scripts/run_fbrac.sh 1 5
```

## Acknowledgments

This codebase is built on top of [FQL](https://github.com/seohongpark/fql) and [OGBench](https://github.com/seohongpark/ogbench)'s reference implementations.
