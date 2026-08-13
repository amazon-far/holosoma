std_msgs/Header header

# Dense per-joint tracking target in holosoma (URDF/Mujoco 29-DOF) joint order.
# joint_names is self-describing; q[i] / dq[i] correspond to joint_names[i].
string[] joint_names
float32[] q                          # target joint positions (rad)
float32[] dq                         # target joint velocities (rad/s)
geometry_msgs/Quaternion root_quat   # reference-body orientation (xyzw via geometry_msgs)
