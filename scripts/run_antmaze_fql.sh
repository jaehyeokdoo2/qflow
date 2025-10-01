use_value_layer_norm=true
use_actor_layer_norm=false
env_name="antmaze-large-navigate-singletask-task1-v0"

python main.py \
    --env_name=$env_name \
    --agent.q_agg=min \
    --agent.alpha=10 \
    --agent.layer_norm=$use_value_layer_norm \
    --agent.actor_layer_norm=$use_actor_layer_norm \


python -m test.q_analysis \
    --env_name=antmaze-large-navigate-singletask-task1-v0 \
    --agent=agents/fql.py \
    --agent.q_agg=min \
    --agent.alpha=10 \
    --agent.layer_norm=false \
    --agent.actor_layer_norm=true \