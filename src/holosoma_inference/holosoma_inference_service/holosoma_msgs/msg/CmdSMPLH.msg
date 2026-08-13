std_msgs/Header header

# false = no usable tracking this frame; consumers skip
bool valid

string[] joint_names
geometry_msgs/Pose[] joint_poses
