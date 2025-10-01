use_value_layer_norm=true
use_actor_layer_norm=false
env_name="antmaze-large-navigate-singletask-task1-v0"

python main.py \
    --offline_steps=1000000 \
    --env_name=$env_name \
    --agent=agents/rebrac.py \
    --agent.alpha_actor=0.001 \
    --agent.alpha_critic=0.01 \
    --agent.layer_norm=$use_value_layer_norm \
    --agent.actor_layer_norm=$use_actor_layer_norm \

# python -m test.q_analysis \
#     --env_name=antmaze-large-navigate-singletask-task1-v0 \
#     --agent=agents/rebrac.py \
#     --agent.layer_norm=true \
#     --agent.actor_layer_norm=true \