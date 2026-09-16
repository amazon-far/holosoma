# Raw policy output this tick (pre-scale network action). header.stamp matches
# the paired /holosoma/observation so consumers can zip (obs -> action) pairs.
std_msgs/Header header
# Teleop-input stamp of the CmdDense frame this tick consumed; matches the
# paired /holosoma/observation source_stamp. header.stamp is robot time.
builtin_interfaces/Time source_stamp
string wandb_id
float32[] data
