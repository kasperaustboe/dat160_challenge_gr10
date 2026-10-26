# DAT160 Challenge Group 10

La til et utgangspunkt for en prm algoritme samt mulighet for å ha visuelt utseende av prm rett i rviz.
I rviz Add -> ByTopic -> Under "prm_graph" -> trykk på "MarkerArray"

Startes med :

ros2 run rqt_image_view rqt_image_view

ros2 launch multi_robot_challenge_23 rescue_robots_w1.launch.py

ros2 run dat160_challenge_gr10 leader

ros2 run dat160_challenge_gr10 robot_handler
