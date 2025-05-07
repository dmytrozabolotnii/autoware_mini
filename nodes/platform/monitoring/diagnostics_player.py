#!/usr/bin/env python3

import os
import rospy
from diagnostic_msgs.msg import DiagnosticArray

import soundfile as sf
import sounddevice as sd

class DiagnosticsPlayer:
    def __init__(self):
        # Parameters

        # Other initializations
        self.sounds_dir = os.path.join(os.path.dirname(__file__), "sounds", "generated")

        # Publishers

        # Subscribers
        rospy.Subscriber('/diagnostics', DiagnosticArray, self.diagnostics_callback, queue_size=1)
    
    def play_sound(self, sound_file):
        sound_path = os.path.join(self.sounds_dir, f"{sound_file}.wav")
        if os.path.exists(sound_path):
            data, fs = sf.read(sound_path, dtype='float32')
            sd.play(data, fs, blocking=True)
    
    def diagnostics_callback(self, msg):
        #print(len(msg.status))
        pass        
    
    def run(self):
        rospy.spin()
        
if __name__ == '__main__':
    rospy.init_node('diagnostics_player')
    player = DiagnosticsPlayer()
    player.run()