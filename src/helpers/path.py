import math
import shapely
import numpy as np
from scipy.interpolate import interp1d
import shapely.ops
from autoware_mini.msg import Waypoint
from geometry_msgs.msg import Point, Pose
from helpers.geometry import get_heading_between_two_points, get_orientation_from_heading
from helpers.shapely import get_heading_at_distance_along_linestring

class PathWrapper:
    def __init__(self, waypoints, velocities=False, blinkers=False, boundaries=False):

        if len(waypoints) == 1:
            ValueError("PathWrapper - waypoints array must be empty or have more than 1 waypoint ")

        self.waypoints = waypoints
        self._waypoints_xyz = np.array([(waypoint.position.x, waypoint.position.y, waypoint.position.z) for waypoint in self.waypoints])

        self.linestring = shapely.LineString(self._waypoints_xyz)
        shapely.prepare(self.linestring)

        d = np.cumsum(np.sqrt(np.sum(np.diff(self._waypoints_xyz[:, :2], axis=0)**2, axis=1)))
        self._distances = np.insert(d, 0, 0)

        self.left_boundary = None
        self.right_boundary = None

        if velocities:
            v = np.array([waypoint.speed for waypoint in self.waypoints])
            distance_to_velocity_interpolator = interp1d(self._distances, v, kind='linear', bounds_error=False, fill_value=0.0)
            self._distance_to_velocity_interpolator = distance_to_velocity_interpolator

        if blinkers:
            b = np.array([(waypoint.blinker_state) for waypoint in self.waypoints])
            distance_to_blinker_interpolator = interp1d(self._distances, (b).astype(np.float) , kind='previous', bounds_error=False, fill_value=Waypoint.STR_STRAIGHT)
            self._distance_to_blinker_interpolator = distance_to_blinker_interpolator

        if boundaries:
            self._generate_boundaries()

    def get_waypoint_index_at_distance(self, distance, side="left"):
        """
        Get waypoint at a certain distance along the path
        :param distance: distance along the path (m)
        :param side: side to search for the distance
        :return: waypoint
        """

        return np.searchsorted(self._distances, distance, side)

    def _extract_waypoints(self, index_start, index_end, copy=False):
        """
        Extract waypoints from index_start to index_end
        :param index_start: start index
        :param index_end: end index
        :return: waypoints
        """
        waypoints = []

        if copy:
            # for each new waypoint copy only the necessary parts
            for waypoint in self.waypoints[index_start:index_end]:
                new_waypoint = Waypoint(position=waypoint.position,
                                        heading=waypoint.heading,
                                        speed=waypoint.speed, 
                                        left_width=waypoint.left_width,
                                        right_width=waypoint.right_width,
                                        lanechange_state=waypoint.lanechange_state,
                                        blinker_state=waypoint.blinker_state)
                waypoints.append(new_waypoint)
        else:
            waypoints = self.waypoints[index_start:index_end]

        return waypoints
    
    def _generate_boundaries(self):
        if len(self.waypoints) < 2:
            return
        
        elif len(self.waypoints) == 2:
            left_offsets = [self.waypoints[0].left_width, self.waypoints[1].left_width]
            right_offsets = [self.waypoints[0].right_width, self.waypoints[1].right_width]

            left_offset_lines = shapely.offset_curve([self.linestring, self.linestring], left_offsets)
            right_offset_lines = shapely.offset_curve([self.linestring, self.linestring], right_offsets)

            left_boundary_coords = [(left_offset_lines[0].coords[0][0], left_offset_lines[0].coords[0][1], self.waypoints[0].position.z),
                                    (left_offset_lines[1].coords[1][0], left_offset_lines[1].coords[1][1], self.waypoints[1].position.z)]
            right_boundary_coords = [(right_offset_lines[0].coords[0][0], right_offset_lines[0].coords[0][1], self.waypoints[0].position.z),
                                    (right_offset_lines[1].coords[1][0], right_offset_lines[1].coords[0][1], self.waypoints[1].position.z)]
            
        else:
            three_point_lines = []
            left_offsets = []
            right_offsets = []

            # create three-point linestring segments for every waypoint
            for i in range(len(self.waypoints)):
                left_offsets.append(self.waypoints[i].left_width)
                right_offsets.append(-self.waypoints[i].right_width)

                # the first and last waypoints cannot be in the middle
                if i == 0:
                    j = i + 1
                elif i == len(self.waypoints) - 1:
                    j = i - 1
                else:
                    j = i

                three_point_lines.append([[self.waypoints[j-1].position.x, self.waypoints[j-1].position.y],
                                        [self.waypoints[j].position.x, self.waypoints[j].position.y],
                                        [self.waypoints[j+1].position.x, self.waypoints[j+1].position.y]])

            three_point_linestrings = shapely.linestrings(three_point_lines)
            
            left_offset_lines = shapely.offset_curve(three_point_linestrings, left_offsets, join_style="mitre")
            right_offset_lines = shapely.offset_curve(three_point_linestrings, right_offsets, join_style="mitre")

            assert len(three_point_linestrings) == len(left_offsets) == len(right_offsets)

            left_boundary_coords = []
            right_boundary_coords = []
            for i in range(len(three_point_linestrings)):
                z_coord = self.waypoints[i].position.z

                if i == 0: # use the first point of the first segment as the first lane boundary point 
                    j = 0
                elif i == len(three_point_linestrings) - 1: # use the third point of the last segment as the last lane boundary point 
                    j = 2
                else: # take the second point from every other segment
                    j = 1
                    
                left_offset_point = left_offset_lines[i].coords[j]
                right_offset_point = right_offset_lines[i].coords[j]

                left_boundary_coords.append((left_offset_point[0], left_offset_point[1], z_coord))
                right_boundary_coords.append((right_offset_point[0], right_offset_point[1], z_coord))

        left_boundary, right_boundary = shapely.linestrings([left_boundary_coords, right_boundary_coords])
        shapely.prepare(left_boundary)
        shapely.prepare(right_boundary)

        self.left_boundary = left_boundary
        self.right_boundary = right_boundary

    def extract_waypoints(self, distance_start, distance_end, trim=False, copy=False):
        """
        Get waypoints between two distances along the path
        :param distance_start: start distance along the path (m)
        :param distance_end: end distance along the path (m)
        :param trim: trim the waypoints to exact distances
        :return: waypoints
        """

        index_start = self.get_waypoint_index_at_distance(distance_start, side="right")
        index_end = self.get_waypoint_index_at_distance(distance_end, side="left")

        # extend the path by one waypoint backwards
        index_start = max(0, index_start - 1)

        # if indices differ by 1, then 1 waypoint is returnd and a linestring cannot be created. Therefore return empty list instead
        if abs(index_start - index_end) == 1:
            return []
        
        waypoints = self._extract_waypoints(index_start, index_end, copy=copy)        

        if trim:
            # modify start and end of the path by shifting waypoints to exact locations determined by distances
            waypoints[0].speed = float(self._distance_to_velocity_interpolator(distance_start))
            waypoints[0].blinker_state = int(self._distance_to_blinker_interpolator(distance_start))
            start_wp_pose = self.linestring.interpolate(distance_start)
            # z will remain the same
            waypoints[0].position.x = start_wp_pose.x
            waypoints[0].position.y = start_wp_pose.y

            waypoints[-1].speed = float(self._distance_to_velocity_interpolator(distance_end))
            waypoints[-1].blinker_state = int(self._distance_to_blinker_interpolator(distance_end))
            end_wp_pose = self.linestring.interpolate(distance_end)
            # z will remain the same
            waypoints[-1].position.x = end_wp_pose.x
            waypoints[-1].position.y = end_wp_pose.y

        return waypoints

    def get_velocity_at_distance(self, distance):
        """
        Get the target velocity at a certain distance along the path.
        :param distance: distance from the path start (m)
        :return: target velocity
        """
        assert hasattr(self, '_distance_to_velocity_interpolator'), "Velocity interpolator not available, check that path was initialized with velocities=True"
        return float(self._distance_to_velocity_interpolator(distance))

    def get_blinker_state_with_lookahead(self, ego_distance_from_path_start, blinker_lookahead_distance):
        """
        Get blinker state. 
        :param ego_distance_from_path_start: distance from path start (m)
        :param blinker_lookahead_d: distance to look ahead point for blinkers (m)
        :return: LampCmd (l, r) included in VehicleCmd
        """
        assert hasattr(self, '_distance_to_blinker_interpolator'), "Blinker interpolator not available, check that path was initialized with blinkers=True"
        current_pose_blinker_state = int(self._distance_to_blinker_interpolator(ego_distance_from_path_start))

        if current_pose_blinker_state != Waypoint.STR_STRAIGHT:
            return get_blinker_state(current_pose_blinker_state)
        else:
            lookahead_blinker_state = int(self._distance_to_blinker_interpolator(blinker_lookahead_distance))
            return get_blinker_state(lookahead_blinker_state)
        
    def get_blinker_at_distance(self, ego_distance_from_path_start):
        """
        Get blinker steering state. 
        :param ego_distance_from_path_start: distance from path start (m)
        :return: steering state
        """
        assert hasattr(self, '_distance_to_blinker_interpolator'), "Blinker interpolator not available, check that path was initialized with blinkers=True"
        return int(self._distance_to_blinker_interpolator(ego_distance_from_path_start))
    
    def get_elevation_at_distance(self, distance):
        """
        Get the elevation at a certain distance along the path.
        :param distance: distance from the path start (m)
        :return: elevation
        """
        point_location = self.linestring.interpolate(distance)
        return point_location.z

    def get_pose_at_distance(self, distance):
        """
        Get perpendicular pose at a certain distance along the path
        :param distance: distance along the path (m)
        :return: Pose
        """

        # Find the point on the path
        point_location = self.linestring.interpolate(distance)
        # if distance is negative for interpolate it is measured from the end of the linestring in reverse direction

        # point is not at the very beginning of the path
        if distance >= 0.1:
            point_before = self.linestring.interpolate(distance - 0.1)
            heading = get_heading_between_two_points(point_before, point_location)
        # use forward point if distance is negative
        else:
            point_after = self.linestring.interpolate(distance + 0.1)
            heading = get_heading_between_two_points(point_location, point_after)

        pose = Pose(position = Point(x = point_location.x, y = point_location.y, z = point_location.z),
                    orientation = get_orientation_from_heading(heading))

        return pose


    def get_heading_at_distance(self, distance):
        """
        Get heading of the path at a given distance
        :param distance: distance along the path
        :return: heading angle in radians
        """

        return get_heading_at_distance_along_linestring(self.linestring, distance)

    def get_cross_track_error(self, current_position):
        """
        Get cross track error - calc distance from track and get the sign
        :param current_pose: current pose
        :return: cross track error
        """

        current_position = shapely.Point(current_position.x, current_position.y, current_position.z)
        return calculate_cross_track_error(self.linestring, current_position)


def get_blinker_state(steering_state):
    """
    Get blinker state  from WaypointState/steering_state
    :param steering_state: steering state
    :return: LampCmd (l, r) included in VehicleCmd
    """

    if steering_state == Waypoint.STR_LEFT:
        return 1, 0
    elif steering_state == Waypoint.STR_RIGHT:
        return 0, 1
    elif steering_state == Waypoint.STR_STRAIGHT:
        return 0, 0
    else:
        return 0, 0

def calculate_cross_track_error(linestring, position):
    """
    Calculate cross track error - calc distance from track and get the sign
    https://robotics.stackexchange.com/questions/22989/what-is-wrong-with-my-stanley-controller-for-car-steering-control

    :param linsetring: shapely linestring
    :param position: current position
    :return: cross track error
    """

    ego_distance_from_path_start = linestring.project(position)

    # if distance is negative it is measured from the end of the linestring in reverse direction
    pos1 = linestring.interpolate(max(0, ego_distance_from_path_start - 0.1))
    pos2 = linestring.interpolate(ego_distance_from_path_start + 0.1)

    numerator = (pos2.x - pos1.x) * (pos1.y - position.y) - (pos1.x - position.x) * (pos2.y - pos1.y)
    denominator = math.sqrt((pos2.x - pos1.x) ** 2 + (pos2.y - pos1.y) ** 2)

    return numerator / denominator

