# 2D Toy Experiment

A modular implementation of a 2D toy experiment comparing three flow-based RL agents:
- **FQL**: Flow Q-Learning (simple baseline)
- **FBRAC**: Flow-based Behavioral Regularized Actor-Critic
- **QFlow**: Q-guided Flow with intermediate value function

## Project Structure

```
2d_experiment/
├── __init__.py              # Package initialization
├── utils.py                 # Utility functions and constants
├── agents.py                # Agent implementations (train_fql, train_fbrac, train_qflow)
├── plotting.py              # Visualization functions
├── toy_experiment.ipynb     # Main notebook (orchestrates training & evaluation)
└── README.md                # This file
```

## Module Descriptions

### `utils.py`
Contains:
- **Constants**: Domain bounds, dataset size
- **Data generation**: Swiss roll and two spirals distributions
- **Neural network utilities**: MLP initialization, forward passes, Fourier embeddings
- **Evaluation metrics**: Wasserstein distance, Maximum Mean Discrepancy (MMD)

### `agents.py`
Contains three agent implementations:
- `train_fql()` - Simple flow Q-learning with one-step distillation
- `train_fbrac()` - Flow-based actor-critic with gradient backprop through ODE
- `train_qflow()` - Q-guided flow with intermediate value function

Each returns:
- Flow policy parameters
- Critic parameters (and inner/outer for QFlow)
- Training history

### `plotting.py`
Visualization utilities:
- `plot_agent_samples()` - Visualize flow samples
- `plot_target_distribution()` - Show target data
- `plot_comparison()` - Side-by-side comparison of all agents
- `plot_training_curves()` - Training loss curves
- `plot_value_heatmap()` - Value function visualization

### `toy_experiment.ipynb`
Main notebook that:
1. Loads utilities and agents
2. Generates target data (Swiss roll or two spirals)
3. Creates reward signal
4. Trains all three agents
5. Generates samples from trained policies
6. Evaluates using MMD and other metrics
7. Visualizes results
8. Saves outputs to `results/` directory

## Usage

### Prerequisites
```bash
pip install jax jaxlib optax numpy matplotlib tqdm
```

### Run the Experiment

1. **Launch Jupyter**:
   ```bash
   cd 2d_experiment
   jupyter notebook toy_experiment.ipynb
   ```

2. **Run cells sequentially**:
   - Setup
   - Generate target data
   - Train agents (may take 5-10 minutes)
   - Generate samples
   - Visualize results
   - Save outputs

### Configuration

In the notebook, you can modify:
- `DATASET_TYPE`: `'swiss_roll'` or `'two_spirals'`
- `EPOCHS`: Training iterations (default: 2000)
- `LR`: Learning rate (default: 1e-3)
- `HIDDEN`: Hidden layer size (default: 128)
- `FLOW_STEPS`: Integration steps (default: 10)
- `ALPHA_QFLOW`: Guidance coefficient for QFlow (default: 0.5)

## Output

Results are saved to `results/` directory:
```
results/
├── {dataset}_comparison.png        # Side-by-side sample visualization
├── {dataset}_training_curves.png   # Loss curves
├── {dataset}_target.npy            # Target data
├── {dataset}_fql_samples.npy       # FQL samples
├── {dataset}_fbrac_samples.npy     # FBRAC samples
├── {dataset}_qflow_samples.npy     # QFlow samples
└── {dataset}_metrics.json          # Quantitative metrics (MMD)
```

## Key Features

✅ **Modular design** - Easy to reuse components  
✅ **Clean separation of concerns** - Utilities, agents, plotting separate  
✅ **Minimal notebook** - Orchestration only, no implementation details  
✅ **Reproducible** - Fixed random seeds  
✅ **Extensible** - Easy to add new agents or plotting functions  
✅ **Well-documented** - Docstrings and inline comments  

## Notes

- The notebook is designed to be executed sequentially
- Training typically takes 5-10 minutes on CPU
- JAX enables GPU acceleration if available
- Results are deterministic given fixed random seeds
