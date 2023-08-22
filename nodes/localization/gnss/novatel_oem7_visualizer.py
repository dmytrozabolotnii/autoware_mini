#!/usr/bin/env python3

import rospy
import math
from std_msgs.msg import ColorRGBA
from novatel_oem7_msgs.msg import INSPVA, BESTPOS
from jsk_rviz_plugins.msg import OverlayText

INSPVA_STATUS = {
    0: "INS_INACTIVE",
    1: "INS_ALIGNING",
    2: "INS_HIGH_VARIANCE",
    3: "INS_SOLUTION_GOOD",
    6: "INS_SOLUTION_FREE",
    7: "INS_ALIGNMENT_COMPLETE",
    8: "DETERMINING_ORIENTATION",
    9: "WAITING_INITIALPOS",
    10: "WAITING_AZIMUTH",
    11: "INITIALIZING_BIASES",
    12: "MOTION_DETECT",
    14: "WAITING_ALIGNMENTORIENTATION"
    }

BESTPOS_POS_TYPE = {
    0: "NONE",
    1: "FIXEDPOS",
    2: "FIXEDHEIGHT",
    8: "DOPPLER_VELOCITY",
    16: "SINGLE",
    17: "PSRDIFF",
    18: "WAAS",
    19: "PROPAGATED",
    32: "L1_FLOAT",
    34: "NARROW_FLOAT",
    48: "L1_INT",
    49: "WIDE_INT",
    50: "NARROW_INT",
    51: "RTK_DIRECT_INS",
    52: "INS_SBAS",
    53: "INS_PSRSP",
    54: "INS_PSRDIFF",
    55: "INS_RTKFLOAT",
    56: "INS_RTKFIXED",
    68: "PPP_CONVERGING",
    69: "PPP",
    70: "OPERATIONAL",
    71: "WARNING",
    72: "OUT_OF_BOUNDS",
    73: "INS_PPP_CONVERGING",
    74: "INS_PPP",
    77: "PPP_BASIC_CONVERGING",
    78: "PPP_BASIC",
    79: "INS_PPP_BASIC_CONVERGING",
    80: "INS_PPP_BASIC"
    }

BLACK = ColorRGBA(0.0, 0.0, 0.0, 0.8)
GREEN = ColorRGBA(0.0, 1.0, 0.0, 1.0)
YELLOW = ColorRGBA(1.0, 1.0, 0.0, 1.0)
RED = ColorRGBA(1.0, 0.0, 0.0, 1.0)
WHITE = ColorRGBA(1.0, 1.0, 1.0, 1.0)

INS_SOLUTION_GOOD = 3
INS_RTKFIXED = 56

class NovatelOem7Visualizer:
    def __init__(self):

        # Parameters
        self.number_of_satellites_good = rospy.get_param("number_of_satellites_good")
        self.number_of_satellites_bad = rospy.get_param("number_of_satellites_bad")
        self.location_accuracy_stdev_good = rospy.get_param("location_accuracy_stdev_good")
        self.location_accuracy_stdev_bad = rospy.get_param("location_accuracy_stdev_bad")
        self.differential_age_good = rospy.get_param("differential_age_good")
        self.differential_age_bad = rospy.get_param("differential_age_bad")

        # Publishers
        self.gnss_general_pub = rospy.Publisher('gnss_general', OverlayText, queue_size=1)
        self.inspva_status_pub = rospy.Publisher('gnss_inspva_status', OverlayText, queue_size=1)
        self.bestpos_pos_type_pub = rospy.Publisher('gnss_bestpos_pos_type', OverlayText, queue_size=1)
        self.bestpos_num_sol_svs_pub = rospy.Publisher('gnss_bestpos_num_sol_svs', OverlayText, queue_size=1)
        self.bestpos_loc_stdev_pub = rospy.Publisher('gnss_bestpos_loc_stdev', OverlayText, queue_size=1)
        self.bestpos_diff_age_pub = rospy.Publisher('gnss_bestpos_diff_age', OverlayText, queue_size=1)

        # Subscribers
        rospy.Subscriber('/novatel/oem7/inspva', INSPVA, self.inspva_callback, queue_size=1)
        rospy.Subscriber('/novatel/oem7/bestpos', BESTPOS, self.bestpos_callback, queue_size=1)

        # Internal parameters
        self.global_top = 315
        self.global_left = 10
        self.global_width = 266
        self.global_height = 17
        self.text_size = 10


    def inspva_callback(self, msg):

        fg_color = WHITE
        if msg.status.status != INS_SOLUTION_GOOD:
            fg_color = YELLOW

        if msg.status.status not in INSPVA_STATUS:
            text = "Unknown INS status: {}".format(msg.status.status)
        else:
            text = "INS status: " + INSPVA_STATUS[msg.status.status]

        inspva_status = OverlayText()
        inspva_status.text = text
        inspva_status.top = self.global_top + 20
        inspva_status.left = self.global_left
        inspva_status.width = self.global_width
        inspva_status.height = self.global_height
        inspva_status.text_size = self.text_size
        inspva_status.fg_color = fg_color
        inspva_status.bg_color = BLACK

        self.inspva_status_pub.publish(inspva_status)

    def bestpos_callback(self, msg):
        
        ################# bestpos_pos_type
        fg_color = WHITE
        if msg.pos_type.type != INS_RTKFIXED:
            fg_color = YELLOW

        if msg.pos_type.type not in BESTPOS_POS_TYPE:
            post_type_string = "Unknown position type: {}".format(msg.pos_type.type)
        else:
            post_type_string = "Position type: " + BESTPOS_POS_TYPE[msg.pos_type.type]
        
        bestpos_pos_type = OverlayText()
        bestpos_pos_type.text = post_type_string
        bestpos_pos_type.top = self.global_top + 20 + 1 * self.global_height
        bestpos_pos_type.left = self.global_left
        bestpos_pos_type.width = self.global_width
        bestpos_pos_type.height = self.global_height
        bestpos_pos_type.text_size = self.text_size
        bestpos_pos_type.fg_color = fg_color
        bestpos_pos_type.bg_color = BLACK

        self.bestpos_pos_type_pub.publish(bestpos_pos_type)

        ################# num_sol_svs
        fg_color = WHITE
        if msg.num_sol_svs < self.number_of_satellites_good:
            fg_color = YELLOW
        if msg.num_sol_svs < self.number_of_satellites_bad:
            fg_color = RED

        num_sol_svs = OverlayText()
        num_sol_svs.text = "Num. satellites: {}".format(msg.num_sol_svs)
        num_sol_svs.top = self.global_top + 20 + 2 * self.global_height
        num_sol_svs.left = self.global_left
        num_sol_svs.width = self.global_width
        num_sol_svs.height = self.global_height
        num_sol_svs.text_size = 9
        num_sol_svs.fg_color = fg_color
        num_sol_svs.bg_color = BLACK

        self.bestpos_num_sol_svs_pub.publish(num_sol_svs)

        ################# loc_stdev
        location_stdev = math.sqrt(msg.lat_stdev**2 + msg.lon_stdev**2)

        fg_color = WHITE
        if location_stdev > self.location_accuracy_stdev_good:
            fg_color = YELLOW
        if location_stdev > self.location_accuracy_stdev_bad:
            fg_color = RED

        loc_stdev = OverlayText()
        loc_stdev.text = "Location stdev: {:.2f} m".format(location_stdev)
        loc_stdev.top = self.global_top + 20 + 3 * self.global_height
        loc_stdev.left = self.global_left
        loc_stdev.width = self.global_width
        loc_stdev.height = self.global_height
        loc_stdev.text_size = 9
        loc_stdev.fg_color = fg_color
        loc_stdev.bg_color = BLACK

        self.bestpos_loc_stdev_pub.publish(loc_stdev)

        ################# diff_age
        fg_color = WHITE
        if msg.diff_age > self.differential_age_good:
            fg_color = YELLOW
        if msg.diff_age > self.differential_age_bad:
            fg_color = RED

        diff_age = OverlayText()
        diff_age.text = "Differential age: {:.2f} s".format(msg.diff_age)
        diff_age.top = self.global_top + 20 + 4 * self.global_height
        diff_age.left = self.global_left
        diff_age.width = self.global_width
        diff_age.height = 28
        diff_age.text_size = 9
        diff_age.fg_color = fg_color
        diff_age.bg_color = BLACK

        self.bestpos_diff_age_pub.publish(diff_age)


        if msg.pos_type.type == INS_RTKFIXED and location_stdev < self.location_accuracy_stdev_good and msg.diff_age < self.differential_age_bad and msg.num_sol_svs > self.number_of_satellites_bad:
            gnss_general_text = "GNSS: OK"
            fg_color = WHITE
        else:
            gnss_general_text = "GNSS: Localization warning"
            fg_color = YELLOW

        gnss_general = OverlayText()
        gnss_general.text = gnss_general_text
        gnss_general.top = self.global_top
        gnss_general.left = self.global_left
        gnss_general.width = self.global_width
        gnss_general.height = 20
        gnss_general.text_size = 11
        gnss_general.fg_color = fg_color
        gnss_general.bg_color = BLACK

        self.gnss_general_pub.publish(gnss_general)


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('novatel_oem7_visualizer', log_level=rospy.INFO)
    node = NovatelOem7Visualizer()
    node.run()
