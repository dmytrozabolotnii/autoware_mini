#!/usr/bin/env python3

import math
import rospy
import shapely

from autoware_mini.msg import DetectedObjectArray
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Header, ColorRGBA

from helpers.shapely import get_polygon_width
from helpers.geometry import get_orientation_from_heading

class DetectedObjectsVisualizer:
    def __init__(self):

        self.use_object_width = rospy.get_param('/planning/use_object_width')
        self.published_ids = set()

        self.markers_pub = rospy.Publisher('detected_objects_markers', MarkerArray, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('detected_objects', DetectedObjectArray, self.objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)

        rospy.loginfo("%s - initialized", rospy.get_name())

    def objects_callback(self, msg):
        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = msg.header.frame_id

        new_published_ids = set()
        markers = MarkerArray()
        for obj in msg.objects:
            # centroid
            marker = Marker(header=header)
            marker.ns = 'centroid'
            marker.id = obj.id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose = obj.pose
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
            marker.color = obj.color
            markers.markers.append(marker)
            
            # bounding box
            marker = Marker(header=header)
            marker.ns = 'bounding_box'
            marker.id = obj.id
            marker.type = marker.LINE_STRIP
            marker.action = marker.ADD
            marker.pose = obj.pose
            marker.scale.x = 0.1
            marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)
            half_length = obj.dimensions.x / 2.0
            half_width = obj.dimensions.y / 2.0
            marker.points = [
                Point(-half_length, -half_width, 0.0),
                Point(-half_length, half_width, 0.0),
                Point(half_length, half_width, 0.0),
                Point(half_length, -half_width, 0.0),
                Point(-half_length, -half_width, 0.0),
            ]
            markers.markers.append(marker)

            # convex hull
            if len(obj.convex_hull.points) > 0:
                marker = Marker(header=header)
                marker.ns = 'convex_hull'
                marker.id = obj.id
                marker.type = marker.LINE_STRIP
                marker.action = marker.ADD
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.1
                marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.8)
                marker.points = [Point(p.x, p.y, p.z) for p in obj.convex_hull.points]
                marker.points.append(marker.points[0])
                markers.markers.append(marker)

            # speed arrow
            marker = Marker(header=header)
            marker.ns = 'speed'
            marker.id = obj.id
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.position = obj.pose.position
            yaw = math.atan2(obj.velocity.y, obj.velocity.x)
            marker.pose.orientation = get_orientation_from_heading(yaw)
            marker.scale.x = max(math.sqrt(obj.velocity.x**2 + obj.velocity.y**2), 0.01)
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color = ColorRGBA(1.0, 1.0, 0.0, 1.0)
            markers.markers.append(marker)

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

            # text
            marker = Marker(header=header)
            marker.ns = 'text'
            marker.id = obj.id
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            marker.pose.position = Point(obj.pose.position.x, obj.pose.position.y, obj.pose.position.z + 1.0)
            marker.scale.z = 0.5
            marker.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
            marker.text = "%s %d (%d km/h)" % (obj.label, obj.id, math.sqrt(obj.velocity.x**2 + obj.velocity.y**2 + obj.velocity.z**2) * 3.6)
            markers.markers.append(marker)

            new_published_ids.add(obj.id)

        # delete ids not published any more
        delete_ids = self.published_ids - new_published_ids
        for id in delete_ids:
            marker = Marker(header=header)
            marker.ns = 'centroid'
            marker.id = id
            marker.action = marker.DELETE
            markers.markers.append(marker)

            marker = Marker(header=header)
            marker.ns = 'bounding_box'
            marker.id = id
            marker.action = marker.DELETE
            markers.markers.append(marker)

            marker = Marker(header=header)
            marker.ns = 'convex_hull'
            marker.id = id
            marker.action = marker.DELETE
            markers.markers.append(marker)

            marker = Marker(header=header)
            marker.ns = 'speed'
            marker.id = id
            marker.action = marker.DELETE
            markers.markers.append(marker)

            marker = Marker(header=header)
            marker.ns = 'candidate_trajectories'
            marker.id = id
            marker.action = marker.DELETE
            markers.markers.append(marker)

            marker = Marker(header=header)
            marker.ns = 'text'
            marker.id = id
            marker.action = marker.DELETE
            markers.markers.append(marker)
        self.published_ids = new_published_ids

        # publish markers
        self.markers_pub.publish(markers)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('detected_objects_visualizer', log_level=rospy.INFO)
    node = DetectedObjectsVisualizer()
    node.run()
