use_value_layer_norm=true
use_actor_layer_norm=false
env_name="antmaze-large-diverse-v2"

python main.py \
    --offline_steps=500000 \
    --env_name=$env_name \
    --agent=agents/rebrac.py \
    --agent.alpha_actor=0.003 \
    --agent.alpha_critic=0.01 \
    --agent.layer_norm=$use_value_layer_norm \
    --agent.actor_layer_norm=$use_actor_layer_norm \