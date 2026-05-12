"""
QFlow hyperparameter configuration.
Structured similar to reproduce.py agent_params.
"""

# Hyperparameters for QFlow agent per domain
# Based on environments specified in scripts/run_qflow.sh

QFLOW_PARAMS = {
    "antmaze-large-navigate": dict(
        alpha=0.2,
        use_time_embed=True,
        time_embed_dim=16,
        q_agg='mean',
        actor_loss_type="grad",
        discount=0.99,
    ),
    "antmaze-giant-navigate": dict(
        alpha=0.2,
        use_time_embed=True,
        time_embed_dim=16,
        q_agg='min',
        actor_loss_type="grad",
        discount=0.995,
    ),
    "antsoccer-arena-navigate": dict(
        alpha=0.5,
        use_time_embed=True,
        time_embed_dim=16,
        q_agg='mean',
        actor_loss_type="grad",
        discount=0.995,
    ),
    "humanoidmaze-medium-navigate": dict(
        alpha=1,
        use_time_embed=True,
        time_embed_dim=16,
        q_agg='mean',
        actor_loss_type="grad",
        discount=0.995,
    ),
    "humanoidmaze-large-navigate": dict(
        alpha=1,
        use_time_embed=True,
        time_embed_dim=16,
        q_agg='mean',
        actor_loss_type="grad",
        discount=0.995,
    ),
    "cube-single-play": dict(
        alpha=5,
        use_time_embed=True,
        time_embed_dim=16,
        q_agg='mean',
        actor_loss_type="grad",
        discount=0.99,
    ),
    "cube-double-play": dict(
        alpha=2,
        use_time_embed=True,
        time_embed_dim=16,
        q_agg='mean',
        actor_loss_type="grad",
        discount=0.99,
    ),
    "scene-play": dict(
        alpha=5,
        use_time_embed=True,
        time_embed_dim=16,
        q_agg='mean',
        actor_loss_type="grad",
        discount=0.99,
    ),
    "puzzle-3x3-play": dict(
        alpha=20,
        use_time_embed=True,
        time_embed_dim=16,
        q_agg='min',
        actor_loss_type="grad",
        discount=0.99,
    ),
    "puzzle-4x4-play": dict(
        alpha=20,
        use_time_embed=True,
        time_embed_dim=16,
        q_agg='min',
        actor_loss_type="grad",
        discount=0.99,
    ),
    # D4RL antmaze tasks
    "antmaze-umaze": dict(
        alpha=0.2,
        use_time_embed=True,
        time_embed_dim=16,
        q_agg='mean',
        actor_loss_type="grad",
        discount=0.99,
    ),
    "antmaze-umaze-diverse": dict(
        alpha=0.2,
        use_time_embed=True,
        time_embed_dim=16,
        q_agg='mean',
        actor_loss_type="grad",
        discount=0.99,
    ),
    "antmaze-medium-play": dict(
        alpha=0.5,
        use_time_embed=True,
        time_embed_dim=16,
        q_agg='mean',
        actor_loss_type="grad",
        discount=0.99,
    ),
    "antmaze-medium-diverse": dict(
        alpha=0.5,
        use_time_embed=True,
        time_embed_dim=16,
        q_agg='mean',
        actor_loss_type="grad",
        discount=0.99,
    ),
    "antmaze-large-play": dict(
        alpha=0.2,
        use_time_embed=True,
        time_embed_dim=16,
        q_agg='mean',
        actor_loss_type="grad",
        discount=0.99,
    ),
    "antmaze-large-diverse": dict(
        alpha=0.2,
        use_time_embed=True,
        time_embed_dim=16,
        q_agg='mean',
        actor_loss_type="grad",
        discount=0.99,
    ),
}


def get_qflow_params(domain):
    """
    Get hyperparameters for a given domain.
    
    Args:
        domain: Domain name (e.g., "cube-triple-play")
        
    Returns:
        Dictionary of hyperparameters for that domain
    """
    return QFLOW_PARAMS.get(domain, {})


def extract_domain_from_env_name(env_name):
    """
    Extract domain name from environment name.
    
    Examples:
        "cube-double-play-singletask-task1-v0" -> "cube-double-play"
        "puzzle-3x3-play-singletask-task1-v0" -> "puzzle-3x3-play"
        "antmaze-large-navigate-singletask-task1-v0" -> "antmaze-large-navigate"
    """
    # Remove task suffix
    if "-singletask-task" in env_name:
        base = env_name.split("-singletask-task")[0]
    elif "-singletask" in env_name:
        base = env_name.split("-singletask")[0]
    else:
        base = env_name
    
    # Remove version suffix (e.g. -v0, -v2) if present
    import re
    base = re.sub(r'-v\d+$', '', base)
    
    return base
