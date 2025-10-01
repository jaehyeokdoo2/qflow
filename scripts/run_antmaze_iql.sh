use_value_layer_norm=true
use_actor_layer_norm=true
env_name="antmaze-large-navigate-singletask-task1-v0"

python main.py \
    --env_name=$env_name \
    --agent=agents/iql.py \
    --agent.alpha=10 \
    --agent.layer_norm=$use_value_layer_norm \
    --agent.actor_layer_norm=$use_actor_layer_norm \