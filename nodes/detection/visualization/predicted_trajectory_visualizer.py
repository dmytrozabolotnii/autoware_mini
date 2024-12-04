#!/usr/bin/env python3

import math
import rospy
import shapely

from autoware_mini.msg import DetectedObjectArray
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Header, ColorRGBA

from helpers.shapely import get_polygon_width

class PredictedTrajectoryVisualizer:
    def __init__(self):

        self.use_object_width = rospy.get_param('/planning/use_object_width')
        self.published_ids = set()

        self.markers_pub = rospy.Publisher('predicted_objects_markers', MarkerArray, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('predicted_objects', DetectedObjectArray, self.objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)

        rospy.loginfo("%s - initialized", rospy.get_name())

    def objects_callback(self, msg):
        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = msg.header.frame_id

        new_published_ids = set()
        markers = MarkerArray()
        for obj in msg.objects:

            # candidate trajectories
            # if len(obj.candidate_trajectories.paths) > 0:
            # extract and visualize object width - used in object detection
            marker = Marker(header=header)
            marker.ns = 'candidate_trajectories'
            marker.id = obj.id
            marker.type = marker.LINE_LIST
            if len(obj.candidate_trajectories.paths) == 0:
                marker.action = marker.DELETE
            else:
                marker.action = marker.ADD
                marker.pose.orientation.w = 1.0
                marker.color = ColorRGBA(1.0, 1.0, 0.0, 0.5)
                if self.use_object_width:
                    object_polygon = shapely.Polygon([(p.x, p.y) for p in obj.convex_hull.points])
                    object_heading = math.degrees(math.atan2(obj.velocity.y, obj.velocity.x))
                    marker.scale.x = get_polygon_width(object_polygon, object_heading)
                else:
                    marker.scale.x = 0.2
                # visualize possible multiple trajectories
                for lane in obj.candidate_trajectories.paths:
                    for i in range(len(lane.waypoints) - 1):
                        p1 = lane.waypoints[i].position
                        p2 = lane.waypoints[i + 1].position
                        marker.points.append(Point(p1.x, p1.y, p1.z))
                        marker.points.append(Point(p2.x, p2.y, p2.z))
            markers.markers.append(marker)

            new_published_ids.add(obj.id)

        # delete ids not published any more
        delete_ids = self.published_ids - new_published_ids
        for id in delete_ids:
            marker = Marker(header=header)
            marker.ns = 'candidate_trajectories'
            marker.id = id
            marker.action = marker.DELETE
            markers.markers.append(marker)

        self.published_ids = new_published_ids

        # publish markers
        self.markers_pub.publish(markers)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('predicted_trajectory_visualizer', log_level=rospy.INFO)
    node = PredictedTrajectoryVisualizer()
    node.run()
