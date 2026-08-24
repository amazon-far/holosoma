# Everything needed to reproduce the obs -> action -> command chain offline.
# Published latched (transient-local) once per policy activation; re-published
# on policy swap so wandb_id always matches the live Observation/Action stream.


std_msgs/Header header
string wandb_id

string observation_terms
# JSON schema of the observation vector: {schema_version, dtype, wandb_id,
# groups: {<group>: {total_dim, terms: [{name, start, dim, history_len,
# total_dim}]}}}. Mirrors the concat order in BasePolicy._update_obs_history.

float32 policy_action_scale
float32[] default_dof_angles
string[] dof_names
