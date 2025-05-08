#!/usr/bin/env python3

import os, csv
import rospy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from queue import Queue
from threading import Thread

import soundfile as sf
import sounddevice as sd

class DiagnosticsPlayer:
    def __init__(self):
        # Parameters
        self.monitoring_conf_path = rospy.get_param('~monitoring_conf_path')
        
        # Other initializations
        self.sounds_dir = os.path.join(os.path.dirname(__file__), "sounds", "generated")
        self.monitored_components = {}
        with open(self.monitoring_conf_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                component = row['component']
                self.monitored_components[component.strip()] = {
                    'freq': DiagnosticStatus.OK,
                    'delay': DiagnosticStatus.OK,
                }
        self.soundfile_queue = Queue()
                
        # Publishers

        # Subscribers
        rospy.Subscriber('/diagnostics', DiagnosticArray, self.diagnostics_callback, queue_size=1)
    
    def play_sound(self, sound_file):
        sound_path = os.path.join(self.sounds_dir, f"{sound_file}")
        if os.path.exists(sound_path):
            data, fs = sf.read(sound_path, dtype='float32')
            sd.play(data, fs, blocking=True)
    
    def diagnostics_callback(self, msg):
        # Array will always have exactly one element
        message = msg.status[0]
        if message.name in self.monitored_components:
            #1st slice is for frequency
            #2nd slice is for delay
            message_freq_slice, message_delay_slice = message.message.split(",")
            
            # Get the status level of frequency and delay from the message
            def get_status_level(slice):
                if "error" in slice:
                    return DiagnosticStatus.ERROR
                elif "warning" in slice:
                    return DiagnosticStatus.WARN
                else:
                    return DiagnosticStatus.OK
            freq_status = get_status_level(message_freq_slice)
            delay_status = get_status_level(message_delay_slice)
            
            # If the statuses have changed, play the corresponding sound (put file in queue)
            def get_status_message(status):
                if status == DiagnosticStatus.ERROR:
                    return "error"
                elif status == DiagnosticStatus.WARN:
                    return "warning"
                else:
                    return "ok"
            if freq_status != self.monitored_components[message.name]['freq']:
                sound_file = f"{message.name.lower()}_frequency_{get_status_message(freq_status)}.wav"
                self.soundfile_queue.put(sound_file)
                self.monitored_components[message.name]['freq'] = freq_status
            if delay_status != self.monitored_components[message.name]['delay']:
                sound_file = f"{message.name.lower()}_delay_{get_status_message(freq_status)}.wav"
                self.soundfile_queue.put(sound_file)
                self.monitored_components[message.name]['delay'] = delay_status
            
    def poll_soundfile_queue(self):
        while not rospy.is_shutdown():
            while not self.soundfile_queue.empty():
                sound_file = self.soundfile_queue.get()
                self.play_sound(sound_file)
    
    def run(self):
        Thread(target=rospy.spin, daemon=True).start()
        self.poll_soundfile_queue()
        
if __name__ == '__main__':
    rospy.init_node('diagnostics_player')
    player = DiagnosticsPlayer()
    player.run()