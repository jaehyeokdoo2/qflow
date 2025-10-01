## Agent aummary

### Antmaze Large 

#### FBRAC
##### Tuned Hyperparameters:
```yaml
use_value_layer_norm=true
use_actor_layer_norm=false
actor_weight_decay=0.0
alpha=30
q_agg=mean
reward_discount=0.995
```

* Achieves upto success rate of 90 consistently across the tasks and seeds

##### Reported Hyperparameters (FQL):
```yaml
use_value_layer_norm=true
use_actor_layer_norm=false
alpha=10
q_agg=mean
reward_discount=0.99
```


#### FQL

##### Tuned Hyperparameters:
```yaml
```

* Performance summary

##### Reported Hyperparameters (FQL):
```yaml
```


#### IQL

##### Tuned Hyperparameters:
```yaml
use_value_layer_norm=true
use_actor_layer_norm=false
alpha=1
reward_discount=0.99
expectile=0.9

# Loss configs
use_mse=false 
normalize_q_loss=false
actor_weight_decay=0.0
actor_jacobian_reg=0.0
actor_hessian_reg=0.0
weight_jacobian_reg=false
```

* Couldn't see much improved with weight decay (L2), Jacobian reg, and Hessian reg.
* A bit of bump in humanoid (higher dimensional tasks) with high Jacobian reg, but should be investigated (especially, the steep performance oscillation)
* Around 80-90 across the tasks and seeds

##### Reported Hyperparameters (FQL):
```yaml
```