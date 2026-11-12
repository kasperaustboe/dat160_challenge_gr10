# DAT160 Challenge Group 10

La til et utgangspunkt for en prm algoritme samt mulighet for å ha visuelt utseende av prm rett i rviz.
I rviz Add -> ByTopic -> Under "prm_graph" -> trykk på "MarkerArray"

Startes med :

ros2 run rqt_image_view rqt_image_view

ros2 launch dat160_challenge_gr10 rescue_robots_w1.launch.py

ros2 run dat160_challenge_gr10 robot_handler

ros2 run dat160_challenge_gr10 leader --ros-args -p use_sim_time:=true


# NYE START KOMMANDOER SLIK AT DEN STARTER MED KUN 2

# Starter Gazebo ++

ros2 launch dat160_challenge_gr10 rescue_robots_w1.launch.py

# Starter Leader, controller osv

ros2 launch dat160_challenge_gr10 controllers.launch.py