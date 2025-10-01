use_value_layer_norm=true
use_actor_layer_norm=true
env_name="antmaze-large-navigate-singletask-task1-v0"

python main.py \
    --offline_steps=1000000 \
    --env_name=$env_name \
    --agent=agents/ifql_modified.py \
    --agent.layer_norm=$use_value_layer_norm \
    --agent.actor_layer_norm=$use_actor_layer_norm \

# python -m test.q_analysis \
#     --env_name=antmaze-large-navigate-singletask-task1-v0 \
#     --agent=agents/ifql.py \
#     --agent.num_samples=32 \
#     --agent.layer_norm=true \
#     --agent.actor_layer_norm=true \