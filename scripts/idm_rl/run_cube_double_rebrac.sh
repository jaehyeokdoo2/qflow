IDM_path="/mnt/nas/jaehyeok/fql/exp/idm/Debug/idm_cube-double-play-singletask-task2-v0_sd000_20250827_003147"
IDM_restore_epoch=390600

use_value_layer_norm=true
use_actor_layer_norm=false
env_name="cube-double-play-singletask-task2-v0"

python main_idm_rl.py \
    --env_name=$env_name \
    --agent=agents/rebrac_idm.py \
    --idm_path=$IDM_path \
    --idm_restore_epoch=$IDM_restore_epoch \
    --agent.alpha_actor=1 \
    --agent.alpha_critic=0 \
    --agent.layer_norm=$use_value_layer_norm \
    --agent.actor_layer_norm=$use_actor_layer_norm \
