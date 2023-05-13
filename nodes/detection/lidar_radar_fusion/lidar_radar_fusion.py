#!/usr/bin/env python3

import numpy as np
import rospy
import cv2
import message_filters
from autoware_msgs.msg import DetectedObjectArray
from std_msgs.msg import ColorRGBA
from math import sqrt

GREEN = ColorRGBA(0.0, 1.0, 0.0, 0.8)

class LidarRadarFusion:
    def __init__(self):

        # Parameters
        self.matching_distance = rospy.get_param("~matching_distance") # radius threshold value around lidar centroid for a radar object to be considered matched
        self.radar_speed_threshold = rospy.get_param("~radar_speed_threshold")  # Threshold for filtering out stationary objects based on speed

        # Subscribers and tf listeners
        radar_detections_sub = message_filters.Subscriber('radar/detected_objects', DetectedObjectArray, queue_size=1)
        lidar_detections_sub = message_filters.Subscriber('lidar/detected_objects', DetectedObjectArray, queue_size=1)

        # Sync
        ts = message_filters.ApproximateTimeSynchronizer([radar_detections_sub, lidar_detections_sub], queue_size=15, slop=0.05)
        ts.registerCallback(self.radar_lidar_data_callback)
        # Publishers
        self.detected_object_array_pub = rospy.Publisher('detected_objects', DetectedObjectArray, queue_size=1)

        rospy.loginfo(rospy.get_name().split('/')[-1] + " - Initialized")

    def radar_lidar_data_callback(self, radar_detections, lidar_detections):
        """
        radar_detections: DetectedObjectArray
        lidar_detections: DetectedObjectArray
        publish: DetectedObjectArray
        """
        final_objects = self.match_and_fuse_detections(lidar_detections, radar_detections)
        self.detected_object_array_pub.publish(final_objects)

    def match_and_fuse_detections(self, lidar_detections, radar_detections):

        """
        radar_prepared: radar object dictionary containing DetectedObjects and their ids
        lidar_prepared: lidar object dictionary containing DetectedObjects and their ids
        return: matches and within_hull_radar_ids
        """
        final_detections = DetectedObjectArray()
        final_detections.header = lidar_detections.header

        for lidar_detection in lidar_detections.objects:
            # Extracting lidar hull points
            lidar_hull = np.array([[hull_point.x, hull_point.y] for hull_point in lidar_detection.convex_hull.polygon.points], dtype=np.float32) #unpacking geometry_msg/Point32 to float values

            min_radar_speed = np.inf
            matched_radar_detection = None
            # For each radar object check if its centroid lies within the convex hull of the lidar object
            # or if the distance between the two centroids is smaller than the self.maching_distance param
            for radar_detection in radar_detections.objects:
                radar_object_centroid = (radar_detection.pose.position.x, radar_detection.pose.position.y)
                distance = self.compute_distance(lidar_detection, radar_detection)

                # calculate norm of radar detection's speed
                radar_speed = self.compute_norm(radar_detection)
                # check if the radar object falls within(+1) or one the edge(0) of the lidar hull. Side note: outside of hull = -1
                is_within_hull = cv2.pointPolygonTest(lidar_hull, radar_object_centroid, measureDist=False) >= 0

                # Add all moving radar objects falling outside lidar hulls  to final objects
                if not is_within_hull and radar_speed >= self.radar_speed_threshold:
                    final_detections.objects.append(radar_detection)

                # check if matched
                if is_within_hull or distance < self.matching_distance:
                    # match the radar object with the lowest speed
                    if radar_speed < min_radar_speed:
                        min_radar_speed = radar_speed
                        matched_radar_detection = radar_detection

            if matched_radar_detection is None:
                final_detections.objects.append(lidar_detection)
            else:
                fused_detection = self.fuse_detections(lidar_detection, matched_radar_detection)
                final_detections.objects.append(fused_detection)

        return final_detections

    def fuse_detections(self, lidar_detection, radar_detection):
        """
        lidar_detection: DetectedObject
        radar_detection: DetectedObject
        return: lidar_detection - fused detection of type DetectedObject
        """
        lidar_detection.velocity = radar_detection.velocity
        lidar_detection.velocity_reliable = True
        lidar_detection.acceleration = radar_detection.acceleration
        lidar_detection.acceleration_reliable = True
        lidar_detection.color = GREEN
        return lidar_detection

    @staticmethod
    def compute_distance(obj1, obj2):
        return sqrt((obj1.pose.position.x - obj2.pose.position.x)**2 + (obj1.pose.position.y - obj2.pose.position.y)**2)

    @staticmethod
    def compute_norm(obj):
        return sqrt(obj.velocity.linear.x**2 + obj.velocity.linear.y**2 + obj.velocity.linear.z**2)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('radar_lidar_fusion', anonymous=True, log_level=rospy.INFO)
    node = LidarRadarFusion()
    node.run()
