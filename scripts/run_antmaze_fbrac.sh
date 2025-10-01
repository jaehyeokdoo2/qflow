use_value_layer_norm=true
use_actor_layer_norm=false
# env_name="antmaze-large-navigate-singletask-task1-v0"
env_name="humanoidmaze-medium-navigate-singletask-task2-v0"
offline_steps=1000000
eval_interval=50000

python main.py \
    --offline_steps=$offline_steps \
    --eval_interval=$eval_interval \
    --env_name=$env_name \
    --agent=agents/fbrac.py \
    --agent.q_agg=min \
    --agent.alpha=30 \
    --agent.normalize_q_loss=true \
    --agent.layer_norm=$use_value_layer_norm \
    --agent.actor_layer_norm=$use_actor_layer_norm \