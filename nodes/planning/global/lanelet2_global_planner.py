#!/usr/bin/env python3
import math
import copy
import itertools

import rospy
import lanelet2
import numpy as np
from lanelet2.core import BasicPoint2d
from lanelet2.geometry import to2D, findWithin2d, length2d, distance as lanelet2_distance
from shapely import distance, Point as ShapelyPoint
from scipy.interpolate import BPoly

from geometry_msgs.msg import PoseStamped, TwistStamped, Point
from autoware_msgs.msg import Lane, Waypoint, WaypointState
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Empty, EmptyResponse
from visualization_msgs.msg import MarkerArray, Marker

from helpers.geometry import get_heading_between_two_points, get_orientation_from_heading, \
    get_distance_between_two_points_2d, get_point_position_2d, angle_between_three_points
from helpers.lanelet2 import load_lanelet2_map
from helpers.path import Path

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
        self.lanelet_search_radius = rospy.get_param("~lanelet_search_radius")
        self.lane_change_length = rospy.get_param("~lane_change_length")

        lanelet2_map_name = rospy.get_param("~lanelet2_map_name")
        coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")

        # Internal variables
        self.lanelet_candidates = []
        self.current_location = None
        self.current_speed = None
        self.start_point = None
        self.goal_point = None

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

        if self.current_location is None:
            # TODO handle if current_pose gets lost at later stage - see current_pose_callback
            rospy.logwarn("%s - current_pose not available", rospy.get_name())
            return

        # Using current pose as start point
        if self.start_point is None:
            start_point = ShapelyPoint(self.current_location.x, self.current_location.y)
            # Get nearest lanelets to start point
            start_lanelet_candidates = findWithin2d(self.lanelet2_map.laneletLayer, BasicPoint2d(start_point.x, start_point.y), self.lanelet_search_radius)
            # If no lanelet found near start point, return
            if len(start_lanelet_candidates) == 0:
                rospy.logerr("%s - no lanelet found near start point", rospy.get_name())
                return
            # Extract lanelet objects from candidates
            start_lanelet_candidates = [start_lanelet[1] for start_lanelet in start_lanelet_candidates]
            lanelet_candidates = [start_lanelet_candidates]
        else:
            start_point = self.start_point
            lanelet_candidates = copy.copy(self.lanelet_candidates)
        
        new_goal = ShapelyPoint(msg.pose.position.x, msg.pose.position.y)
        # Get nearest lanelets to goal point
        goal_lanelet_candidates = findWithin2d(self.lanelet2_map.laneletLayer, BasicPoint2d(new_goal.x, new_goal.y), self.lanelet_search_radius)
        # If no lanelet found near goal point, return
        if len(goal_lanelet_candidates) == 0:
            rospy.logerr("%s - no lanelet found near goal point", rospy.get_name())
            return
        # Extract lanelet objects from candidates
        goal_lanelet_candidates = [goal_lanelet[1] for goal_lanelet in goal_lanelet_candidates]
        # Add current goal candidates to lanelet candidates list
        lanelet_candidates.append(goal_lanelet_candidates)

        # Find shortest path and shortest route
        path, route = self.get_shortest_path_with_route(lanelet_candidates)
        if path is None:
            rospy.logerr("%s - no route found, try new goal!", rospy.get_name())
            return

        # Publish target lanelets for visualization
        start_lanelet = path[0]
        goal_lanelet = path[-1]
        self.publish_target_lanelets(start_lanelet, goal_lanelet)
        
        global_path = Path(self.convert_to_waypoints(path, route), velocities=True, blinkers=True)
        
        #waypoints = self.convert_to_waypoints(path, route)

        # Find distance to start and goal waypoints
        start_point_distance = global_path.linestring.project(start_point)
        new_goal_point_distance = global_path.linestring.project(new_goal)
        # interpolate point coordinates
        start_on_path = global_path.linestring.interpolate(start_point_distance)
        new_goal_on_path = global_path.linestring.interpolate(new_goal_point_distance)

        if distance(start_on_path, start_point) > self.distance_to_centerline_limit:
            rospy.logerr("%s - start point too far from centerline", rospy.get_name())
            return

        if distance(new_goal_on_path, new_goal) > self.distance_to_centerline_limit:
            rospy.logerr("%s - goal point too far from centerline", rospy.get_name())
            return

        if start_lanelet.id == goal_lanelet.id and start_point_distance > new_goal_point_distance:
            rospy.logerr("%s - goal point can't be on the same lanelet before start point", rospy.get_name())
            return

        # If there is only one goal candidate, we can fix the preceding lanelets to be the best found route
        if len(goal_lanelet_candidates) == 1:
            lanelet_candidates = [[lanelet] for lanelet in route]

        "====="
        # trim the global path
        trimmed_waypoints = global_path.extract_waypoints(start_point_distance, new_goal_point_distance, trim=True, copy=True)

        # calculate lane changes
        lane_change_waypoints = self.create_lane_change_paths(trimmed_waypoints)
        if lane_change_waypoints is None:
            rospy.logerr("%s - calculated path contained an impossible lane change", rospy.get_name())
            return
        
        self.waypoints += lane_change_waypoints

        # update member variables
        self.goal_point = new_goal_on_path
        self.start_point = start_point
        self.lanelet_candidates = lanelet_candidates
        rospy.logdebug("Lanelet candidates: " + str(list(map(len, lanelet_candidates))))

        # publish the global path
        waypoints = global_path.extract_waypoints(start_point_distance, new_goal_point_distance, trim=True)
        self.publish_waypoints(waypoints)
        rospy.loginfo("%s - global path published", rospy.get_name())


    def current_pose_callback(self, msg):
        self.current_location = ShapelyPoint(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)

        if self.goal_point != None:
            d = distance(self.current_location, self.goal_point)
            if d < self.distance_to_goal_limit and self.current_speed < self.ego_vehicle_stopped_speed_limit:
                self.goal_point = None
                self.start_point = None
                self.lanelet_candidates = []
                self.publish_waypoints([])
                rospy.loginfo("%s - goal reached, clearing path!", rospy.get_name())

    def current_velocity_callback(self, msg):
        self.current_speed = msg.twist.linear.x

    def cancel_route_callback(self, msg):
        self.goal_point = None
        self.start_point = None
        self.lanelet_candidates = []
        self.publish_waypoints([])
        rospy.loginfo("%s - route cancelled!", rospy.get_name())
        return EmptyResponse()
    
    def get_shortest_path_with_route(self, lanelet_candidates):
        shortest_path = None
        shortest_route = None
        shortest_distance = math.inf
        possible_routes = list(itertools.product(*lanelet_candidates))
        for possible_route in possible_routes:
            path = self.graph.shortestPathWithVia(possible_route[0], possible_route[1:-1], possible_route[-1], 0, self.lane_change)
            if path is not None:
                path_length = sum(map(length2d, path))
                if path_length < shortest_distance:
                    shortest_distance = path_length
                    shortest_route = possible_route
                    shortest_path = path

        return shortest_path, shortest_route

    def convert_to_waypoints(self, lanelet_sequence, route):
        waypoints = []

        last_lanelet = False
        lane_change_end_lanelets = []

        for i, lanelet in enumerate(lanelet_sequence):
            if i == len(lanelet_sequence)-1:
                last_lanelet = True

            lane_change_state = 0
            lane_change = not last_lanelet and lanelet.centerline[-1] != lanelet_sequence[i+1].centerline[0]
            additional_lanelets = [] # additional lanelets in case the current lanelet is too short for a lane change

            # Check for lane change
            if lane_change:
                # Skip the middle lanelets if there are multiple lane changes in succession
                if i > 0 and lanelet.centerline[0] != lanelet_sequence[i-1].centerline[-1]:
                    continue

                n = 1
                
                # Check if there is also a lane change on the following lanelets
                next_lane_change = False
                for ii in range(i+1, len(lanelet_sequence)-1):
                    if lanelet_sequence[ii].centerline[-1] != lanelet_sequence[ii+1].centerline[0]:
                        next_lane_change = True
                        n += 1
                    else:
                        break

                lanelets_crossover_dist = get_distance_between_two_points_2d(lanelet.centerline[0], lanelet_sequence[i+n].centerline[-1])
                lane_change_end_lanelets.append(i+n)
                lane_change_state = 1
                j = 2
                
                # Try extending the lanelet with following lanelets if the current lanelet is too short for a lane change 
                while lanelets_crossover_dist < self.lane_change_length and not next_lane_change:

                    if len(lanelet_sequence) <= i+j: # No following lanelets
                        return None
                    
                    additional_lanelet = self.extend_lane_change_lanelet(lanelet, lanelet_sequence[i+2], route)

                    if additional_lanelet is None: # No following adjacent lanelets found
                        break
                    
                    additional_lanelets.append(additional_lanelet)
                    lane_change_end_lanelets.append(i+j)

                    lanelets_crossover_dist = get_distance_between_two_points_2d(lanelet.centerline[0], lanelet_sequence[i+j].centerline[-1])
                    j += 1

            if i in lane_change_end_lanelets:
                lane_change_state = 2
            
            # Loop over centerline points
            for idx in range(0, len(lanelet.centerline)):
                if not last_lanelet and idx == len(lanelet.centerline)-1:
                    # Skip last point on every lanelet (except last), because it is the same as the first point of the following lanelet
                    break

                point = lanelet.centerline[idx]

                if last_lanelet and idx == len(lanelet.centerline)-1:
                    # Use heading of previous point - last point of last lanelet has no following point
                    waypoint = self.create_waypoint(point, lanelet, lanelet.centerline[idx-1], lanelet.centerline[idx], lane_change=lane_change_state)
                else:
                    waypoint = self.create_waypoint(point, lanelet, lanelet.centerline[idx], lanelet.centerline[idx+1], lane_change=lane_change_state)

                waypoints.append(waypoint)

            for additional_lanelet in additional_lanelets:
                for idx, point in enumerate(additional_lanelet.centerline):
                    if idx == len(lanelet.centerline)-1:
                        break

                    waypoint = self.create_waypoint(point, additional_lanelet, additional_lanelet.centerline[idx], 
                                                    additional_lanelet.centerline[idx+1], lane_change=1)
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
        marker.header.frame_id = self.output_frame
        marker.header.stamp = rospy.Time.now()
        marker.action = Marker.ADD
        marker.type = Marker.POINTS
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        return marker
    
    def create_lane_change_paths(self, waypoints):
        lane_change_idxs = []
        
        lane_change_start_point = None
        other_point = None
        start_idx = None

        for idx, waypoint in enumerate(waypoints):

            # Find waypoint where the lane change starts
            if waypoint.wpstate.lanechange_state == 1 and lane_change_start_point is None:
                lane_change_start_point = ShapelyPoint(waypoint.pose.pose.position.x, waypoint.pose.pose.position.y)
                start_idx = idx
                if idx < len(waypoints)-1 and waypoints[idx+1].wpstate.lanechange_state == 1:
                    other_point = ShapelyPoint(waypoints[idx+1].pose.pose.position.x, waypoints[idx+1].pose.pose.position.y)
                else:
                    # If there is no following points on the start lanelet, use a previous point that is moved 
                    # directly opposite of the lane change start point
                    o_x = 2 * lane_change_start_point.x - waypoints[idx-1].pose.pose.position.x
                    o_y = 2 * lane_change_start_point.y - waypoints[idx-1].pose.pose.position.y
                    other_point = ShapelyPoint(o_x, o_y)

            # Find waypoint where the lane change ends
            elif waypoint.wpstate.lanechange_state == 2 and lane_change_start_point is not None:
                current_point = ShapelyPoint(waypoint.pose.pose.position.x, waypoint.pose.pose.position.y)
                d = get_distance_between_two_points_2d(lane_change_start_point, current_point)
                a = angle_between_three_points(other_point, lane_change_start_point, current_point)

                if abs(a) < np.pi/2 and d > self.lane_change_length:
                    lane_change_idxs.append((start_idx, idx))
                    lane_change_start_point = None
                    other_point = None
                    start_idx = None

            # No lane change end waypoint was found
            elif waypoint.wpstate.lanechange_state == 0 and lane_change_start_point is not None:
                return None
            
        if start_idx is not None:
            return None

        lane_change_idxs.reverse()
        for s, e in lane_change_idxs:
            # get two other points to calculate direction vectors
            if s == 0:
                waypoint1 = waypoints[s+1]
                d1 = 1
            else:
                waypoint1 = waypoints[s-1]
                d1 = -1

            if e == len(waypoints) - 1:
                waypoint2 = waypoints[e-1]
                d2 = 1
            else:
                waypoint2 = waypoints[e+1]
                d2 = -1

            lane_change_waypoints = self.calculate_lane_change_spline(waypoints[s], waypoints[e], waypoint1, waypoint2, (d1, d2))
            waypoints = waypoints[:s] + lane_change_waypoints + waypoints[e+1:]

        return waypoints
    
    def calculate_lane_change_spline(self, start_waypoint, end_waypoint, waypoint1, waypoint2, directions):
        waypoints = []
        dir1, dir2 = directions

        ##################################################################
        # Calculate Bezier curve control points p0, p1, p2, p3
        ##################################################################

        p0 = np.array([start_waypoint.pose.pose.position.x, start_waypoint.pose.pose.position.y])

        p3 = np.array([end_waypoint.pose.pose.position.x, end_waypoint.pose.pose.position.y])

        dx1 = (waypoint1.pose.pose.position.x - p0[0])*dir1
        dy1 = (waypoint1.pose.pose.position.y - p0[1])*dir1

        vec_length1 = np.sqrt(dx1**2 + dy1**2)
        dx1 /= vec_length1
        dy1 /= vec_length1

        scaled_x1 = dx1 * self.lane_change_length / 3
        scaled_y1 = dy1 * self.lane_change_length / 3

        p1 = np.array([p0[0] + scaled_x1, p0[1] + scaled_y1])

        dx2 = (waypoint2.pose.pose.position.x - p3[0])*dir2
        dy2 = (waypoint2.pose.pose.position.y - p3[1])*dir2

        vec_length2 = np.sqrt(dx2**2 + dy2**2)
        dx2 /= vec_length2
        dy2 /= vec_length2

        scaled_x2 = dx2 * self.lane_change_length / 3
        scaled_y2 = dy2 * self.lane_change_length / 3

        p2 = np.array([p3[0] + scaled_x2, p3[1] + scaled_y2])

        control_points = np.array([p0, p1, p2, p3])

        x = control_points[:, 0]
        y = control_points[:, 1]

        # Coefficients
        c_x = np.array([x]).T
        c_y = np.array([y]).T

        # Breakpoints
        t = np.array([0, 1])

        # Create BPoly objects for x and y coordinates
        bpoly_x = BPoly(c_x, t)
        bpoly_y = BPoly(c_y, t)

        # Generate 10 waypoint coordinates on the Bezier curve
        t_wp = np.linspace(0, 1, 10)
        x_wp = bpoly_x(t_wp)
        y_wp = bpoly_y(t_wp)

        bezier_points = np.array([x_wp, y_wp]).T

        ##################################################################
        # Create lane change waypoints
        ##################################################################

        speed = min(start_waypoint.twist.twist.linear.x, end_waypoint.twist.twist.linear.x)

        lw = min(start_waypoint.dtlane.lw, end_waypoint.dtlane.lw)
        rw = min(start_waypoint.dtlane.rw, end_waypoint.dtlane.rw)

        z_s = start_waypoint.pose.pose.position.z
        z_e = end_waypoint.pose.pose.position.z

        # Determine lane change direction for blinker
        other_point = ShapelyPoint([waypoint1.pose.pose.position.x, waypoint1.pose.pose.position.y])
        if dir1 == 1:
            pos_val = get_point_position_2d(ShapelyPoint(p0), other_point, ShapelyPoint(p3))
        else:
            pos_val = get_point_position_2d(other_point, ShapelyPoint(p0), ShapelyPoint(p3))

        if pos_val < 0:
            blinker = WaypointState.STR_RIGHT
        elif pos_val > 0:
            blinker = WaypointState.STR_LEFT
        else:
            rospy.logwarn("%s - lane change direction not determined", rospy.get_name())
            blinker = WaypointState.STR_STRAIGHT

        # Calculate new heading for the start waypoint
        start_heading = get_heading_between_two_points(ShapelyPoint(p0), ShapelyPoint(bezier_points[0]))
        start_waypoint.pose.pose.orientation = get_orientation_from_heading(start_heading)

        start_waypoint.wpstate.steering_state = blinker
        waypoints.append(start_waypoint)

        for i in range(len(bezier_points)):
            point = ShapelyPoint(bezier_points[i])
            if i == len(bezier_points) - 1:
                next_point = ShapelyPoint(p3)
            else:
                next_point = ShapelyPoint(bezier_points[i+1])

            point_z = z_s + i*(z_e - z_s)/(len(bezier_points) - 1)
            
            coords = (point.x, point.y, point_z)
            waypoint = self.create_waypoint(coords=coords, blinker=blinker, p1=point, p2=next_point, 
                                        speed=speed, lw=lw, rw=rw)
            
            waypoints.append(waypoint)

        waypoints.append(end_waypoint)

        return waypoints
    
    def extend_lane_change_lanelet(self, lanelet, adjacent_lanelet, route):
        following_rels = route.followingRelations(lanelet)
        left_rels = route.leftRelations(adjacent_lanelet)
        right_rels = route.rightRelations(adjacent_lanelet)

        for following_rel in following_rels:
            for left_rel in left_rels:
                if following_rel.lanelet == left_rel.lanelet:
                    return following_rel.lanelet
                
            for right_rel in right_rels:
                if following_rel.lanelet == right_rel.lanelet:
                    return following_rel.lanelet

        return None

    def create_waypoint(self, point=None, lanelet=None, p1=None, p2=None, coords=None,
                        blinker=None, speed=None, lw=None, rw=None, lane_change=0):
        
        if blinker is None:
            if 'turn_direction' in lanelet.attributes:
                blinker = LANELET_TURN_DIRECTION_TO_WAYPOINT_STATE_MAP[lanelet.attributes['turn_direction']]
            else:
                blinker = WaypointState.STR_STRAIGHT

        if speed is None:
            speed = self.speed_limit / 3.6
            if 'speed_limit' in lanelet.attributes:
                speed = min(speed, float(lanelet.attributes['speed_limit']) / 3.6)
            if 'speed_ref' in lanelet.attributes:
                speed = min(speed, float(lanelet.attributes['speed_ref']) / 3.6)

        waypoint = Waypoint()
        
        if coords is not None:
            x, y, z = coords
        else:
            x, y, z = point.x, point.y, point.z

        waypoint.pose.pose.position.x = x
        waypoint.pose.pose.position.y = y
        waypoint.pose.pose.position.z = z
        waypoint.wpstate.steering_state = blinker
        waypoint.wpstate.lanechange_state = lane_change

        # calculate quaternion for orientation
        heading = get_heading_between_two_points(p1, p2)
        waypoint.pose.pose.orientation = get_orientation_from_heading(heading)

        waypoint.twist.twist.linear.x = speed

        if lw is None:
            waypoint.dtlane.lw = lanelet2_distance(point, lanelet.leftBound)
        else:
            waypoint.dtlane.lw = lw

        if rw is None:
            waypoint.dtlane.rw = lanelet2_distance(point, lanelet.rightBound)
        else:
            waypoint.dtlane.rw = rw

        return waypoint

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('lanelet2_global_planner', log_level=rospy.INFO)
    node = Lanelet2GlobalPlanner()
    node.run()