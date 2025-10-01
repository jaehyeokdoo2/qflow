use_value_layer_norm=true
use_actor_layer_norm=false
env_name="antmaze-giant-navigate-singletask-task1-v0"

python main.py \
    --env_name=$env_name \
    --agent=agents/rebrac.py \
    --agent.alpha_actor=0.003 \
    --agent.alpha_critic=0.01 \
    --agent.layer_norm=$use_value_layer_norm \
    --agent.actor_layer_norm=$use_actor_layer_norm \