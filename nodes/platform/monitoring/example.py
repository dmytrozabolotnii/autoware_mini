#!/usr/bin/env python3
from __future__ import print_function

import rospy
import yaml
import wave
import pyaudio
import rospkg
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from threading import Thread


class diagnostics_parser:
    SOUND_PATH = rospkg.RosPack().get_path('autoware_ut') + "/nodes/diagnostics_parser/sounds/"
    config = None
    diagnostics_pub = None
    pyaudio = pyaudio.PyAudio()
    playing = False
    sound_count_trigger = 2
    current_errors = []
    error_count = {}
    first_message = {}

    def __init__(self):
        self.config = yaml.safe_load(open(rospkg.RosPack().get_path('autoware_ut') + "/config/diagnostics/diagnostics_parser.yaml", 'r'))
        rospy.loginfo(self.__class__.__name__ + " - node started")
        rospy.Subscriber('/unprocessed_diagnostics', DiagnosticArray, self.process_diagnostics_status, queue_size=1)
        self.diagnostics_pub = rospy.Publisher('/diagnostics', DiagnosticArray,  queue_size=5)

    def process_diagnostics_status(self, diagnostic_array):
        """
        :type diagnostic_array: DiagnosticArray
        """
        diagnostic_array.status = self.process_status_array(diagnostic_array.status)

        diagnostic_array.header.stamp = rospy.Time.now()
        self.diagnostics_pub.publish(diagnostic_array)

    def process_status_array(self, status_array):
        for i in range(len(status_array)):
            if status_array[i].name in self.config.keys():

                if status_array[i].name not in self.error_count:
                    self.error_count[status_array[i].name] = {}

                if status_array[i].name not in self.first_message:
                    self.first_message[status_array[i].name] = True

                status_array[i] = self.process_status(
                    status_array[i],
                    self.config[status_array[i].name]['tests'],
                    status_array[i].name,
                    self.config[status_array[i].name]['nominal_sound']
                )
        return status_array

    def process_status(self, status, config, device_name, nominal_sound):
        """
        :type status: DiagnosticStatus
        :type config: dict
        :type device_name: str
        :type nominal_sound: str
        :return:
        """
        valid = True
        message = "Nominal"

        # We only count errors that trigger the notif
        errors_in_last_status = 0
        for key in self.error_count[device_name].keys():
            if self.error_count[device_name][key] >= self.sound_count_trigger:
                errors_in_last_status += 1

        for status_value in status.values:

            if self.playing:
                break

            if status_value.key not in config:
                continue

            if self.is_valid(status_value.value, config[status_value.key]):
                self.process_valid_gps_fix_status(status_value, device_name)
            else:
                valid, message = self.process_invalid_gps_fix_status(status_value, config, device_name)

        if len(self.error_count[device_name].keys()) == 0 and (errors_in_last_status > 0 or self.first_message[device_name]) and not self.playing:
            self.play_audio_async(nominal_sound)
            self.first_message[device_name] = False

        status.level = 0 if valid else 1
        status.message = message
        return status

    def process_invalid_gps_fix_status(self, status, config, device_name):
        valid = False
        message = config[status.key]['message']
        rospy.logwarn(self.__class__.__name__ + " - " + config[status.key]['message'])

        if status.key in self.error_count[device_name].keys():
            self.error_count[device_name][status.key] += 1
        else:
            self.error_count[device_name][status.key] = 1

        # Playing sound from config
        if 'sound' in config[status.key] and not self.playing \
                and (
                self.error_count[device_name][status.key] == self.sound_count_trigger
                or "critical" in config[status.key] and self.get_correct_type(config[status.key]['critical'], 'bool')
        ):
            self.play_audio_async(config[status.key]['sound'])

        return valid, message

    def process_valid_gps_fix_status(self, status, device_name):
        if status.key in self.error_count[device_name].keys():
            del(self.error_count[device_name][status.key])

    def is_valid(self, value, config):
        if "exactly" in config.keys():
            return self.get_correct_type(value, config['type']) == self.get_correct_type(config['exactly'], config['type'])
        elif "above" in config.keys():
            return self.get_correct_type(value, config['type']) > self.get_correct_type(config['above'], config['type'])
        elif "below" in config.keys():
            return self.get_correct_type(value, config['type']) < self.get_correct_type(config['below'], config['type'])
        elif "in" in config.keys():
            assert isinstance(config['in'], list), "Not a list"
            correct_list = [self.get_correct_type(item, config['type']) for item in config['in']]
            return self.get_correct_type(value, config['type']) in correct_list
        elif "not_bitwise_and" in config.keys():
            return not bool(self.get_correct_type(value, config['type']) & self.get_correct_type(config['not_bitwise_and'], config['type']))
        else:
            raise Exception("Must contain a condition")

    def get_correct_type(self, value, type):
        if type == "string":
            return str(value)
        elif type == "int":
            return int(value)
        elif type == "float":
            return float(value)
        elif type == "bool":
            # Different rules here because of how YAML integrates bool
            if isinstance(value, str):
                if value in ['yes', 'true', 'True', 'on']:
                    return True
                elif value in ['no', 'false', 'False', 'off']:
                    return False
                raise Exception("'%s' is not a correct boolean value" % value)
            return bool(value)
        raise Exception("This type '%s' is not valid" % type)

    def play_audio_async(self, filename):
        """
        Allows to play sound asynchronously
        :type filename: str
        :return:
        """
        t = Thread(target=self.play_audio, args=(filename,))
        t.start()

    def play_audio(self, filename):
        """
        Allows to play sound
        :type filename: str
        :return:
        """
        self.playing = True
        audio_file = wave.open(self.SOUND_PATH + filename, "rb")

        stream = self.pyaudio.open(
            format=self.pyaudio.get_format_from_width(audio_file.getsampwidth()),
            channels=audio_file.getnchannels(),
            rate=audio_file.getframerate(),
            output=True
        )
        data = audio_file.readframes(audio_file.getframerate())

        while data:
            stream.write(data)
            data = audio_file.readframes(audio_file.getframerate())

        stream.stop_stream()
        stream.close()
        self.playing = False
        return

    @staticmethod
    def run():
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('gps_diagnostics_parser', anonymous=True, log_level=rospy.INFO)
    node = diagnostics_parser()
    node.run()
