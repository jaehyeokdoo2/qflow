IDM_path=/mnt/nas/jaehyeok/fql/exp/idm/Debug/idm_antmaze-large-diverse-v2_sd000_20250826_204435
IDM_restore_epoch=390200

use_value_layer_norm=true
use_actor_layer_norm=false
env_name="antmaze-large-diverse-v2"

python main_idm_rl.py \
    --offline_steps=500000 \
    --env_name=$env_name \
    --agent=agents/fql_idm.py \
    --idm_path=$IDM_path \
    --idm_restore_epoch=$IDM_restore_epoch \
    --agent.aug_loss_weight=0.1 \
    --agent.aug_discount=1 \
    --agent.aug_std=0.01 \
    --agent.q_agg=mean \
    --agent.alpha=3 \
    --agent.layer_norm=$use_value_layer_norm \
    --agent.actor_layer_norm=$use_actor_layer_norm \