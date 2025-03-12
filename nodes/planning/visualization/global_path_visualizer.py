#!/usr/bin/env python3

import rospy
import shapely

from autoware_mini.msg import Path, Waypoint
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point32, Polygon, PolygonStamped
from jsk_recognition_msgs.msg import PolygonArray

from helpers.geometry import get_orientation_from_heading, split_line_fixed_length

class GlobalPathVisualizer:
    def __init__(self):

        # Publishers
        self.global_path_markers_pub = rospy.Publisher('global_path_markers', MarkerArray, queue_size=10, latch=True, tcp_nodelay=True)
        self.global_path_polygons = rospy.Publisher('global_path_polygons', PolygonArray, queue_size=10, latch=True, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('global_path', Path, self.global_path_callback, queue_size=None, tcp_nodelay=True)

    def global_path_callback(self, path):
        marker_array = MarkerArray()

        poly_array = PolygonArray()
        poly_array.header.frame_id = path.header.frame_id
        poly_array.header.stamp = rospy.Time.now()

        if len(path.waypoints) == 0:
            # create marker_array to delete all visualization markers
            marker = Marker()
            marker.header.frame_id = path.header.frame_id
            marker.action = Marker.DELETEALL
            marker_array.markers.append(marker)

        else:
            # Pose arrows
            for i, waypoint in enumerate(path.waypoints):

                # color the arrows based on the waypoint steering_flag (blinker)
                if waypoint.blinker_state == Waypoint.STR_LEFT:
                    color = ColorRGBA(1.0, 0.0, 0.0, 1.0)
                elif waypoint.blinker_state == Waypoint.STR_RIGHT:
                    color = ColorRGBA(0.0, 0.0, 1.0, 1.0)
                else:
                    color = ColorRGBA(0.0, 1.0, 0.0, 1.0)

                marker = Marker()
                marker.header.frame_id = path.header.frame_id
                marker.header.stamp = rospy.Time.now()
                marker.ns = "Waypoint pose"
                marker.id = i
                marker.type = marker.ARROW
                marker.action = marker.ADD
                marker.pose.position = waypoint.position
                marker.pose.orientation = get_orientation_from_heading(waypoint.heading)
                marker.scale.x = 0.4
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                marker.color = color
                marker_array.markers.append(marker)

            # velocity labels
            for i, waypoint in enumerate(path.waypoints):
                marker = Marker()
                marker.header.frame_id = path.header.frame_id
                marker.header.stamp = rospy.Time.now()
                marker.ns = "Velocity label"
                marker.id = i
                marker.type = marker.TEXT_VIEW_FACING
                marker.action = marker.ADD
                marker.pose.position = waypoint.position
                marker.pose.orientation = get_orientation_from_heading(waypoint.heading)
                marker.scale.z = 0.5
                marker.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
                marker.text = str(round(waypoint.speed * 3.6, 1))
                marker_array.markers.append(marker)

            # Split the path into segments, create a buffer (polygon) around the segments and convert them to polygons
            linestring = shapely.linestrings([[p.position.x, p.position.y, p.position.z] for p in path.waypoints])
            linestring = linestring.simplify(0.05)
            
            segments = split_line_fixed_length(linestring, 100)

            for segment in segments:
                seg_buffer = segment.buffer(0.75, cap_style="flat")

                # Extract z coordinates from linestring for each triangle point
                seg_points = shapely.points(seg_buffer.exterior.coords)
                seg_distances = linestring.line_locate_point(seg_points)
                seg_points_on_linestring = linestring.interpolate(seg_distances)

                poly_stamped = PolygonStamped()
                poly_stamped.header.frame_id = path.header.frame_id
                poly_stamped.header.stamp = rospy.Time.now()

                polygon_points = [Point32(p[0], p[1], seg_points_on_linestring[i].z + 0.1) for i, p in enumerate(seg_buffer.exterior.coords[:-1])]
                poly_stamped.polygon = Polygon(points = polygon_points)
                poly_array.polygons.append(poly_stamped)

        self.global_path_markers_pub.publish(marker_array)
        self.global_path_polygons.publish(poly_array)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('global_path_visualizer', log_level=rospy.INFO)
    node = GlobalPathVisualizer()
    node.run()