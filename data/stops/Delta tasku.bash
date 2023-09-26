rostopic pub --once /move_base_simple/goal geometry_msgs/PoseStamped  "header:
  stamp:
    secs: $(date +%s)
    nsecs: $(date +%N)
  frame_id: 'map'
pose:
  position:
    x: -0.4822743295226246
    y: -2.4188089473173022
    z: 0.0
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 0.0
"
