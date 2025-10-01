use_value_layer_norm=true
use_actor_layer_norm=false
# env_name="antmaze-large-navigate-singletask-task2-v0"
env_name="antmaze-giant-navigate-singletask-task2-v0"
discount=0.995

python main.py \
    --seed=0 \
    --offline_steps=1000000 \
    --env_name=$env_name \
    --agent=agents/rebrac.py \
    --agent.discount=$discount \
    --agent.alpha_actor=0.003 \
    --agent.alpha_critic=0.01 \
    --agent.layer_norm=$use_value_layer_norm \
    --agent.actor_layer_norm=$use_actor_layer_norm \