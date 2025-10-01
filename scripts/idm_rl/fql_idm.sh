IDM_path=/mnt/nas/jaehyeok/fql/exp/idm/Debug/idm_antmaze-large-navigate-singletask-task2-v0_sd000_20250827_165122
IDM_restore_epoch=390600

# Env
env_name="antmaze-large-navigate-singletask-task2-v0"
offline_steps=1000000
eval_interval=10000

# DA hyperparams
aug_loss_weight=1
aug_discount=1
aug_std=0.01

# FQL hyperparams
use_value_layer_norm=true
use_actor_layer_norm=false
q_agg="min"
alpha=10 # BC coeff
for i in {1..2}; do
    python main_idm_rl.py \
        --seed=$i \
        --offline_steps=$offline_steps \
        --env_name=$env_name \
        --agent=agents/fql_idm.py \
        --idm_path=$IDM_path \
        --idm_restore_epoch=$IDM_restore_epoch \
        --agent.aug_loss_weight=$aug_loss_weight \
        --agent.aug_discount=$aug_discount \
        --agent.aug_std=$aug_std \
        --agent.q_agg=$q_agg \
        --agent.alpha=$alpha \
        --agent.layer_norm=$use_value_layer_norm \
        --agent.actor_layer_norm=$use_actor_layer_norm
done