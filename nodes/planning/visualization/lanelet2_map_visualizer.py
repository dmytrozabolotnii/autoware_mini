#!/usr/bin/env python3

import rospy
import time

from autoware_mini.msg import TrafficLightResultArray
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import ColorRGBA, Int32
from lanelet2.core import BasicPoint2d
from lanelet2.geometry import findWithin2d, distance, to2D

from helpers.lanelet2 import load_lanelet2_map, get_stop_lines_using_subtype
from helpers.geometry import get_distance_between_two_points_2d
from helpers.timer import Timer

# used for traffic lights
RED = ColorRGBA(1.0, 0.0, 0.0, 0.8)
YELLOW = ColorRGBA(1.0, 1.0, 0.0, 0.8)
GREEN = ColorRGBA(0.0, 1.0, 0.0, 0.8)

# colors for other map features
GREY = ColorRGBA(0.4, 0.4, 0.4, 0.6)
ORANGE = ColorRGBA(1.0, 0.5, 0.0, 0.6)
WHITE = ColorRGBA(1.0, 1.0, 1.0, 0.6)
CYAN = ColorRGBA(0.0, 1.0, 1.0, 0.3)
BLUE = ColorRGBA(0.3, 0.3, 1.0, 0.3)
WHITE100 = ColorRGBA(1.0, 1.0, 1.0, 1.0)

TRAFFIC_LIGHT_STATE_TO_MARKER_COLOR = {
    0: RED,     # red and yellow
    1: GREEN,
    2: WHITE
}

LANELET_COLOR_TO_MARKER_COLOR = {
    "red": RED,
    "yellow": YELLOW,
    "green": GREEN,
}

INDEX_TO_MARKER_COLOR = {
    0: RED,
    1: YELLOW,
    2: GREEN,
}

class Lanelet2MapVisualizer:

    def __init__(self):
    
        # Parameters
        lanelet2_map_name = rospy.get_param("~lanelet2_map_name")
        self.local_path_length = rospy.get_param("/planning/local_path_length")

        self.map_extraction_distance = 500
        
        t = Timer()

        self.current_location = None
        self.map_extraction_location = None

        self.loaded_lanelet2_map = load_lanelet2_map(lanelet2_map_name)
        t("load_lanelet2_map")
        self.yield_stop_lines = get_stop_lines_using_subtype(self.loaded_lanelet2_map, subtype=["yield_stop"])
        t("get_stop_lines_using_subtype")

        # Special publishers for stop line markers: traffic_lights and yielding
        self.tfl_stop_line_markers_pub = rospy.Publisher('tfl_stop_line_markers', MarkerArray, queue_size=1, latch=True, tcp_nodelay=True)
        self.yield_stop_line_markers_pub = rospy.Publisher('yield_stop_line_markers', MarkerArray, queue_size=1, latch=True, tcp_nodelay=True)
        self.lanelet2_map_markers_pub = rospy.Publisher('lanelet2_map_markers', MarkerArray, queue_size=1, latch=True, tcp_nodelay=True)

        rospy.Subscriber("/detection/traffic_light_status", TrafficLightResultArray, self.traffic_light_status_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/planning/lets_go', Int32, self.lets_go_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1, tcp_nodelay=True)

        rospy.loginfo("%s - map loaded with %i lanelets and %i regulatory elements from file: %s", rospy.get_name(),
                      len(self.loaded_lanelet2_map.laneletLayer), len(self.loaded_lanelet2_map.regulatoryElementLayer), lanelet2_map_name)
        t("init finished")
        print(t)

    def current_pose_callback(self, msg):
        t = Timer()
        if self.map_extraction_location is None or get_distance_between_two_points_2d(self.map_extraction_location, msg.pose.position) > (self.map_extraction_distance - 2*self.local_path_length):
            self.map_extraction_location = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)

            filtered_lanelets = findWithin2d(self.loaded_lanelet2_map.laneletLayer, self.map_extraction_location, self.map_extraction_distance)
            filtered_linestrings = findWithin2d(self.loaded_lanelet2_map.lineStringLayer, self.map_extraction_location, self.map_extraction_distance)
            filtered_regulatory_elements = []
            for reg_el in self.loaded_lanelet2_map.regulatoryElementLayer:
                if reg_el.attributes["subtype"] == "traffic_light":
                    for line in reg_el.parameters["ref_line"]:
                        if distance(to2D(line), self.map_extraction_location) <= self.map_extraction_distance:
                            filtered_regulatory_elements.append(reg_el)
                            break

            # Visualize different parts of the map
            lanelet_markers = visualize_laneltLayer(filtered_lanelets)
            linestring_markers = visualize_lineStringLayer(filtered_linestrings)
            reg_el_markers = visualize_regulatoryElementLayer(filtered_regulatory_elements)

           # conactenate the MarkerArrays with delete all at front
            marker_array = MarkerArray()
            marker = Marker()
            marker.action = Marker.DELETEALL
            marker_array.markers = [marker] + lanelet_markers.markers + linestring_markers.markers + reg_el_markers.markers

            t("incb visualize_lanelet2_map")

            # create MarkerArray publisher
            self.lanelet2_map_markers_pub.publish(lanelet_markers)
            t("incb publish markers")
            print(t)

    def lets_go_callback(self, msg):
        marker_array = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        marker_array.markers.append(marker)

        if msg.data != -1:
            points = [Point(x=x, y=y, z=z + 0.01) for x, y, z in self.yield_stop_lines[msg.data].coords]
            marker = linestring_to_marker(points, "Yield line", msg.data, GREEN, 0.5, rospy.Time.now())
            marker_array.markers.append(marker)

        self.yield_stop_line_markers_pub.publish(marker_array)

    def traffic_light_status_callback(self, msg):
        marker_array = MarkerArray()
        # delete all previous markers
        marker = Marker()
        marker.action = Marker.DELETEALL
        marker_array.markers.append(marker)

        states = {}
        for result in msg.results:
            # check if we have already outputted the status of this stopline
            if result.stopline_id in states:
                if states[result.stopline_id] != result.recognition_result_str:
                    rospy.logwarn("%s - multiple traffic lights with different states on the same stop line %d: %s != %s", rospy.get_name(), result.stopline_id, states[result.stopline_id], result.recognition_result_str)
                continue

            # fetch the stop line data
            stop_line = self.loaded_lanelet2_map.lineStringLayer.get(result.stopline_id)
            points = [Point(x=p.x, y=p.y, z=p.z + 0.01) for p in stop_line]

            # choose the color of stopline based on the traffic light state
            if result.recognition_result in TRAFFIC_LIGHT_STATE_TO_MARKER_COLOR:
                color = TRAFFIC_LIGHT_STATE_TO_MARKER_COLOR[result.recognition_result]
            else:
                rospy.logwarn("%s - unrecognized traffic light state: %d", rospy.get_name(), result.recognition_result)
                color = WHITE

            # check if string contains "FLASH" string in it
            if "FLASH" in result.recognition_result_str:
                color = ColorRGBA(color.r, color.g, color.b, color.a * get_multiplier())

            # create linestring marker
            stopline_marker = linestring_to_marker(points, "Stop line", stop_line.id, color, 0.5, rospy.Time.now())

            marker_array.markers.append(stopline_marker)

            # create traffic light status marker
            text_marker = text_to_marker(result.recognition_result_str, points, "Status text", stop_line.id, WHITE100, 0.5, rospy.Time.now())
            marker_array.markers.append(text_marker)

            # record the state of this stop line
            states[result.stopline_id] = result.recognition_result_str

        self.tfl_stop_line_markers_pub.publish(marker_array)

    def run(self):
        rospy.spin()


def get_multiplier():
    if time.time() % 1 < 0.5:
        return 0.5
    else:
        return 1.0


def visualize_laneltLayer(filtered_lanelets):

    # Create a MarkerArray
    marker_array = MarkerArray()

    for _, lanelet in filtered_lanelets:

        stamp = rospy.Time.now()

        # TODO bicycle_lane, bus_lane, emergency_lane, parking_lane, pedestrian_lane, sidewalk, special_lane, traffic_island, traffic_lane, traffic_zone, walkway        
        if lanelet.attributes["subtype"] == "road":
        
            # Create markers for the left, right boundary and centerline
            left_boundary_marker = linestring_to_marker(lanelet.leftBound, "Left boundary", lanelet.id, GREY, 0.1, stamp)
            right_boundary_marker = linestring_to_marker(lanelet.rightBound, "Right boundary", lanelet.id, GREY, 0.1, stamp)
            centerline_marker = linestring_to_marker(lanelet.centerline, "Centerline", lanelet.id, CYAN, 1.5, stamp)

            # Add the markers to the MarkerArray
            marker_array.markers.append(left_boundary_marker)
            marker_array.markers.append(right_boundary_marker)
            marker_array.markers.append(centerline_marker)

        elif lanelet.attributes["subtype"] == "crosswalk":

            points = [point for point in lanelet.leftBound]
            points += [point for point in lanelet.rightBound.invert()]
            points.append(lanelet.leftBound[0])

            crosswalk_marker = linestring_to_marker(points, "Crosswalk", lanelet.id, ORANGE, 0.3, stamp)
            marker_array.markers.append(crosswalk_marker)

        elif lanelet.attributes["subtype"] == "bus_lane":
            centerline_marker = linestring_to_marker(lanelet.centerline, "Centerline", lanelet.id, BLUE, 1.5, stamp)
            marker_array.markers.append(centerline_marker)

    return marker_array

def visualize_regulatoryElementLayer(filtered_regulatory_elements):
    
    # Create a MarkerArray
    marker_array = MarkerArray()

    # Iterate over all the regulatory elements
    for reg_el in filtered_regulatory_elements:
        # Check if the regulatory element is a traffic light group
        if reg_el.attributes["subtype"] == "traffic_light":
            stamp = rospy.Time.now()
            # can have several individual traffic lights
            for tfl in reg_el.parameters["refers"]:
                p1 = tfl[0]
                p2 = tfl[1]

                tfl_height = float(tfl.attributes["height"])

                # calculate bulb positions
                bulb_x = (p1.x + p2.x) / 2
                bulb_y = (p1.y + p2.y) / 2
                bulb_z = p1.z + 5*tfl_height/6

                for i in range(3):
                    # Create a marker for the traffic light bulb
                    marker = Marker()
                    marker.header.frame_id = "map"
                    marker.header.stamp = stamp
                    marker.ns = "Traffic lights"
                    marker.id = len(marker_array.markers)
                    marker.type = marker.SPHERE
                    marker.action = marker.ADD
                    marker.scale.x = tfl_height/6
                    marker.scale.y = tfl_height/6
                    marker.scale.z = tfl_height/6
                    marker.color = INDEX_TO_MARKER_COLOR[i]
                    marker.pose.position.x = bulb_x
                    marker.pose.position.y = bulb_y
                    marker.pose.position.z = bulb_z
                    marker.pose.orientation.w = 1.0

                    marker_array.markers.append(marker)
                    bulb_z -= tfl_height/6

        # TODO stop line, yield line, speed limit, etc.

    return marker_array


def visualize_lineStringLayer(filtered_linestrings):

    marker_array = MarkerArray()

    for _, line in filtered_linestrings:
            # if has attributes
            if line.attributes:
                # select stop lines
                if line.attributes["type"] == "stop_line":
                    points = [point for point in line]
                    if "subtype" in line.attributes:
                        if line.attributes["subtype"]=="traffic_light":
                            marker = linestring_to_marker(points, "Traffic light stop lines", line.id, WHITE, 0.5, rospy.Time.now())
                            marker_array.markers.append(marker)
                        elif line.attributes["subtype"]=="yield_stop":
                            marker = linestring_to_marker(points, "Yield stop line", line.id, RED, 0.5, rospy.Time.now())
                            marker_array.markers.append(marker)
                        elif line.attributes["subtype"]=="yield":
                            marker = linestring_to_marker(points, "Yield line", line.id, YELLOW, 0.3, rospy.Time.now())
                            marker_array.markers.append(marker)

    return marker_array


def linestring_to_marker(linestring, namespace, id, color, scale, stamp):
    """
    Creates a Marker from a LineString
    :param linestring: LineString
    :param namespace: Marker namespace
    :param id: Marker id
    :param color: Marker color
    :param stamp: Marker timestamp
    :return: Marker
    """
    # Create a Marker
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = stamp
    marker.ns = namespace
    marker.id = id
    marker.type = marker.LINE_STRIP
    marker.action = marker.ADD
    marker.scale.x = scale
    marker.color = color
    marker.pose.orientation.w = 1.0

    # Add the points to the marker
    for point in linestring:
        marker.points.append(point)

    return marker

def text_to_marker(text, linestring, namespace, id, color, scale, stamp):
    """
    Creates a Marker from a text
    :param text: text
    :param namespace: Marker namespace
    :param id: Marker id
    :param color: Marker color
    :param stamp: Marker timestamp
    :return: Marker
    """
    # Create a Marker
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = stamp
    marker.ns = namespace
    marker.id = id
    marker.type = marker.TEXT_VIEW_FACING
    marker.action = marker.ADD
    marker.scale.z = scale
    marker.color = color
    marker.pose.position.x = (linestring[0].x + linestring[-1].x) / 2.0
    marker.pose.position.y = (linestring[0].y + linestring[-1].y) / 2.0
    marker.pose.position.z = (linestring[0].z + linestring[-1].z) / 2.0
    marker.pose.orientation.w = 1.0
    marker.text = text

    return marker


if __name__ == '__main__':
    rospy.init_node('lanelet2_map_visualizer')
    node = Lanelet2MapVisualizer()
    node.run()