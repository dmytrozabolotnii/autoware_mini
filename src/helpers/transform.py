import numpy as np
from tf.transformations import quaternion_matrix
from geometry_msgs.msg import PointStamped, Vector3Stamped, PoseStamped
from tf2_geometry_msgs import do_transform_point, do_transform_vector3, do_transform_pose

def transform_point(point, transform):
    # to apply a transform we need a point stamped
    point_stamped = PointStamped(point=point)
    return do_transform_point(point_stamped, transform).point

def transform_vector3(vector3, transform):
    # to apply a transform we need a vector3 stamped
    vector3_stamped = Vector3Stamped(vector=vector3)
    return do_transform_vector3(vector3_stamped, transform).vector

def transform_pose(pose, transform):
    # to apply a transform we need a pose stamped
    pose_stamped = PoseStamped(pose=pose)
    return do_transform_pose(pose_stamped, transform).pose

def transform_to_matrix(transform):
    """
    Convert a ROS Transform to a 4x4 transformation matrix.

    :param transform: geometry_msgs/Transform or TransformStamped message
    :return: 4x4 numpy array representing the transformation matrix
    """
    # Extract translation
    translation = [
        transform.translation.x,
        transform.translation.y,
        transform.translation.z,
    ]

    # Extract rotation quaternion
    quaternion = [
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
        transform.rotation.w,
    ]

    # Convert quaternion to a 4x4 rotation matrix
    rotation_matrix = quaternion_matrix(quaternion)

    # Add translation to the rotation matrix
    transformation_matrix = np.identity(4)
    transformation_matrix[:3, :3] = rotation_matrix[:3, :3]
    transformation_matrix[:3, 3] = translation

    return transformation_matrix
