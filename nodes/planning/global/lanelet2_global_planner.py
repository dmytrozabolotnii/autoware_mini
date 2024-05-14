#!/usr/bin/env python3

import rospy
import copy
import lanelet2
import numpy as np
from lanelet2.core import BasicPoint2d
from lanelet2.geometry import to2D, findNearest, distance
from shapely import LineString, Point as ShapelyPoint

from geometry_msgs.msg import PoseStamped, TwistStamped, Point
from autoware_msgs.msg import Lane, Waypoint, WaypointState
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Empty, EmptyResponse
from visualization_msgs.msg import MarkerArray, Marker

from helpers.geometry import get_heading_between_two_points, get_orientation_from_heading
from helpers.lanelet2 import load_lanelet2_map

LANELET_TURN_DIRECTION_TO_WAYPOINT_STATE_MAP = {
    "straight": WaypointState.STR_STRAIGHT,
    "left": WaypointState.STR_LEFT,
    "right": WaypointState.STR_RIGHT
}

RED = ColorRGBA(1.0, 0.0, 0.0, 0.8)
GREEN = ColorRGBA(0.0, 1.0, 0.0, 0.8)

class Lanelet2GlobalPlanner:

    def __init__(self):

        # Parameters
        self.output_frame = rospy.get_param("~output_frame")
        self.distance_to_goal_limit = rospy.get_param("~distance_to_goal_limit")
        self.distance_to_centerline_limit = rospy.get_param("~distance_to_centerline_limit")
        self.speed_limit = rospy.get_param("~speed_limit")
        self.ego_vehicle_stopped_speed_limit = rospy.get_param("~ego_vehicle_stopped_speed_limit")
        self.lane_change = rospy.get_param("~lane_change")

        lanelet2_map_name = rospy.get_param("~lanelet2_map_name")
        coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")

        # Internal variables
        self.current_location = None
        self.current_speed = None
        self.goal_point = None
        self.waypoints = []

        self.lanelet2_map = load_lanelet2_map(lanelet2_map_name, coordinate_transformer, use_custom_origin, utm_origin_lat, utm_origin_lon)

        # traffic rules
        traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                  lanelet2.traffic_rules.Participants.VehicleTaxi)

        # routing graph
        self.graph = lanelet2.routing.RoutingGraph(self.lanelet2_map, traffic_rules)

        # Publishers
        self.waypoints_pub = rospy.Publisher('global_path', Lane, queue_size=10, latch=True, tcp_nodelay=True)
        self.target_lane_pub = rospy.Publisher('target_lane_markers', MarkerArray, queue_size=10, latch=True, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback, queue_size=None, tcp_nodelay=True)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/localization/current_velocity', TwistStamped, self.current_velocity_callback, queue_size=1, tcp_nodelay=True)

        # Services
        rospy.Service('cancel_route', Empty, self.cancel_route_callback)

    def goal_callback(self, msg):
        rospy.loginfo("%s - goal position (%f, %f, %f) orientation (%f, %f, %f, %f) in %s frame", rospy.get_name(),
                    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z,
                    msg.pose.orientation.w, msg.header.frame_id)

        if self.current_location == None:
            # TODO handle if current_pose gets lost at later stage - see current_pose_callback
            rospy.logwarn("%s - current_pose not available", rospy.get_name())
            return
        
        if self.current_speed == None:
            rospy.logwarn("%s - current_speed not available", rospy.get_name())
            return

        # if there is already a goal, use it as start point
        start_point = self.goal_point if self.goal_point else self.current_location
        new_goal = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)

        # Get nearest lanelets
        goal_lanelet = findNearest(self.lanelet2_map.laneletLayer, new_goal, 1)[0][1]
        start_lanelet = findNearest(self.lanelet2_map.laneletLayer, start_point, 1)[0][1]
        self.publish_target_lanelets(start_lanelet, goal_lanelet)

        route = self.graph.getRoute(start_lanelet, goal_lanelet, 0, self.lane_change)
        if route == None:
            rospy.logerr("%s - no route found, try new goal!", rospy.get_name())
            return

        path = route.shortestPath()

        waypoints = self.convert_to_waypoints(path)
        waypoints_xy = np.array([(w.pose.pose.position.x, w.pose.pose.position.y) for w in waypoints])
        waypoint_distances = np.cumsum(np.sqrt(np.sum(np.diff(waypoints_xy, axis=0)**2, axis=1)))
        waypoint_distances = np.insert(waypoint_distances, 0, 0)
        waypoint_linestring = LineString(waypoints_xy)

        # Find distance to start and goal waypoints
        start_point_distance = waypoint_linestring.project(ShapelyPoint(start_point.x, start_point.y))
        new_goal_point_distance = waypoint_linestring.project(ShapelyPoint(new_goal.x, new_goal.y))
        # interpolate point coordinates
        start_on_path = waypoint_linestring.interpolate(start_point_distance)
        new_goal_on_path = waypoint_linestring.interpolate(new_goal_point_distance)

        if distance(BasicPoint2d(start_on_path.x, start_on_path.y), start_point) > self.distance_to_centerline_limit:
            rospy.logerr("%s - start point too far from centerline", rospy.get_name())
            return

        if distance(BasicPoint2d(new_goal_on_path.x, new_goal_on_path.y), new_goal) > self.distance_to_centerline_limit:
            rospy.logerr("%s - goal point too far from centerline", rospy.get_name())
            return

        index_start = np.searchsorted(waypoint_distances, start_point_distance, side='right')
        index_goal = np.searchsorted(waypoint_distances, new_goal_point_distance, side='left')

        # create new start and goal waypoints using deepcopy and index
        start_wp = copy.deepcopy(waypoints[index_start])
        start_wp.pose.pose.position = Point(start_on_path.x, start_on_path.y, start_wp.pose.pose.position.z)
        goal_wp = copy.deepcopy(waypoints[index_goal])
        goal_wp.pose.pose.position = Point(new_goal_on_path.x, new_goal_on_path.y, goal_wp.pose.pose.position.z)

        # put together new global path
        self.waypoints += [start_wp] + waypoints[index_start : index_goal] + [goal_wp]

        if start_lanelet.id == goal_lanelet.id and index_start > index_goal:
            rospy.logerr("%s - goal point can't be on the same lanelet before start point", rospy.get_name())
            return

        # update goal point and add new waypoints to the existing ones
        self.goal_point = BasicPoint2d(goal_wp.pose.pose.position.x, goal_wp.pose.pose.position.y)

        self.publish_waypoints(self.waypoints)
        rospy.loginfo("%s - path published", rospy.get_name())


    def current_pose_callback(self, msg):
        self.current_location = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)

        if self.goal_point != None:
            d = distance(self.current_location, self.goal_point)
            if d < self.distance_to_goal_limit and self.current_speed < self.ego_vehicle_stopped_speed_limit:
                self.waypoints = []
                self.goal_point = None
                self.publish_waypoints(self.waypoints)
                rospy.loginfo("%s - goal reached, clearing path!", rospy.get_name())

    def current_velocity_callback(self, msg):
        self.current_speed = msg.twist.linear.x

    def cancel_route_callback(self, msg):
        self.waypoints = []
        self.goal_point = None
        self.publish_waypoints(self.waypoints)
        rospy.loginfo("%s - route cancelled!", rospy.get_name())
        return EmptyResponse()

    def convert_to_waypoints(self, lanelet_sequence):
        waypoints = []

        last_lanelet = False

        for i , lanelet in enumerate(lanelet_sequence):
            if 'turn_direction' in lanelet.attributes:
                blinker = LANELET_TURN_DIRECTION_TO_WAYPOINT_STATE_MAP[lanelet.attributes['turn_direction']]
            else:
                blinker = WaypointState.STR_STRAIGHT

            if i == len(lanelet_sequence)-1:
                last_lanelet = True

            speed = self.speed_limit / 3.6
            if 'speed_limit' in lanelet.attributes:
                speed = min(speed, float(lanelet.attributes['speed_limit']) / 3.6)
            if 'speed_ref' in lanelet.attributes:
                speed = min(speed, float(lanelet.attributes['speed_ref']) / 3.6)

            # loop over centerline points use enumerate to get index
            for idx, point in enumerate(lanelet.centerline):
                if not last_lanelet and idx == len(lanelet.centerline)-1:
                    # skip last point on every lanelet (except last), because it is the same as the first point of the following lanelet
                    break

                # calculate quaternion for orientation
                if last_lanelet and idx == len(lanelet.centerline)-1:
                    # use heading of previous point - last point of last lanelet has no following point
                    heading = get_heading_between_two_points(lanelet.centerline[idx-1], lanelet.centerline[idx])
                else:
                    heading = get_heading_between_two_points(lanelet.centerline[idx], lanelet.centerline[idx+1])

                waypoint = Waypoint()
                waypoint.pose.pose.position.x = point.x
                waypoint.pose.pose.position.y = point.y
                waypoint.pose.pose.position.z = point.z
                waypoint.wpstate.steering_state = blinker
                waypoint.pose.pose.orientation = get_orientation_from_heading(heading)
                waypoint.twist.twist.linear.x = speed
                waypoint.dtlane.lw = distance(point, lanelet.leftBound)
                waypoint.dtlane.rw = distance(point, lanelet.rightBound)

                waypoints.append(waypoint)

        return waypoints


    def publish_waypoints(self, waypoints):

        lane = Lane()        
        lane.header.frame_id = self.output_frame
        lane.header.stamp = rospy.Time.now()
        lane.waypoints = waypoints
        
        self.waypoints_pub.publish(lane)


    def publish_target_lanelets(self, start_lanelet, goal_lanelet):
        
        marker_array = MarkerArray()

        # create correct ones
        marker = self.create_target_lanelet_marker()
        marker.ns = "start_lanelet"
        marker.color = GREEN
        for point in to2D(start_lanelet.centerline):
            marker.points.append(Point(point.x, point.y, 0.0))
        marker_array.markers.append(marker)

        marker = self.create_target_lanelet_marker()
        marker.ns = "goal_lanelet"
        marker.color = RED
        for point in to2D(goal_lanelet.centerline):
            marker.points.append(Point(point.x, point.y, 0.0))
        marker_array.markers.append(marker)

        self.target_lane_pub.publish(marker_array)
    
    def create_target_lanelet_marker(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.action = Marker.ADD
        marker.type = Marker.POINTS
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        return marker

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('lanelet2_global_planner')
    node = Lanelet2GlobalPlanner()
    node.run()