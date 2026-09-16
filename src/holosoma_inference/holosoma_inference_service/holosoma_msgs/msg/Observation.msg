# Raw policy observation vector fed to the network this tick (actor_obs).
# Slice-by-name layout is described once on /holosoma/policy_metadata; wandb_id
# links this sample to the metadata that describes it (survives policy swaps).
std_msgs/Header header
# Teleop-input stamp of the CmdDense frame this tick consumed (repeats under
# hold-last-frame). Matches the source teleop log; header.stamp is robot time.
builtin_interfaces/Time source_stamp
string wandb_id
float32[] data
