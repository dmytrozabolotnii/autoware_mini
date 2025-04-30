#!/usr/bin/env python3

"""
TODO
* Create yaml file for listing the topics to monitor 
    and their desired frequency and latency ranges.
* Create a launch file to run this node.
* Copy sound files from autoware_ut.
"""

import rospy
import pyaudio, wave
from threading import Thread
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from rosgraph_msgs.msg import TopicStatistics
from rospkg import RosPack

class DiagnosticsMonitor:
    def __init__(self):
        
        # Load configuration
        self.topics_config = None
        
        # Other initializations
        # Audio
        self.pyaudio = pyaudio.PyAudio()
        self.sound_path = RosPack().get_path('autoware_mini') + "/nodes/platform/monitoring/sounds/"
        self.playing = False
        
        # Publishers
        self.diagnostics_pub = rospy.Publisher('/diagnostics', DiagnosticArray, queue_size=5)
        
        # Subscribers
        rospy.Subscriber('/statistics', TopicStatistics, self.topic_statistics_callback, queue_size=1)


    def topic_statistics_callback(self, msg):
        """
        Callback for topic statistics.
        """
        pass
    
    def publish_diagnostic(self, name, level, message):
        """
        Publish diagnostic message.
        """
        diagnostic = DiagnosticStatus()
        diagnostic.name = name
        diagnostic.level = level
        diagnostic.message = message

        diagnostic_array = DiagnosticArray()
        diagnostic_array.status.append(diagnostic)
        diagnostic_array.header.stamp = rospy.Time.now()
        self.diagnostics_pub.publish(diagnostic_array)
    
    def play_audio_async(self, filename):
        """
        Play audio asynchronously.
        """
        if not self.playing:
            Thread(target=self.play_audio, args=(filename,)).start()
    
    def play_audio(self, filename):
        """
        Play audio file.
        """
        self.playing = True
        try:
            audio_file = wave.open(self.sound_path + filename, "rb")
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
        finally:
            self.playing = False
    
    def run(self):
        rospy.spin()

if __name__ == "__main__":
    rospy.init_node('diagnostics_monitor')
    monitor = DiagnosticsMonitor()
    monitor.run()