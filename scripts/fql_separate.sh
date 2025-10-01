# Env
env_name="antmaze-large-navigate-singletask-task1-v0"
# env_name="humanoidmaze-medium-navigate-singletask-task1-v0"
offline_steps=1000000
eval_interval=50000

# FQL hyperparams
use_value_layer_norm=true
use_actor_layer_norm=false
alpha=2.5 # BC coeff
reward_discount=0.99

for i in {0..0}; do
    python main_separate.py \
        --num_actor_updates=4 \
        --num_critic_updates=1 \
        --seed=$i \
        --offline_steps=$offline_steps \
        --eval_interval=$eval_interval \
        --env_name=$env_name \
        --agent=agents/fql_separate.py \
        --agent.discount=$reward_discount \
        --agent.alpha=$alpha \
        --agent.layer_norm=$use_value_layer_norm \
        --agent.actor_layer_norm=$use_actor_layer_norm
done