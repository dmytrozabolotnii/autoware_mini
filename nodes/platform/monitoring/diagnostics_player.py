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
                    'freq': { 
                        'status': DiagnosticStatus.OK,
                        'history': [DiagnosticStatus.OK],
                        'ok_since': -1,
                    },
                    'delay': {
                        'status': DiagnosticStatus.OK,
                        'history': [DiagnosticStatus.OK],
                        'ok_since': -1,
                    }
                }
        self.ok_min_duration = rospy.Duration(2)  # seconds
        self.soundfile_queue = Queue()
        
        self.startup_ignore_duration = rospy.Duration(5)  # seconds
        rospy.sleep(0.5) # Wait for the clock to be set (necessary when playing back bag files)
        self.start_time = rospy.Time.now()

        # Subscribers
        rospy.Subscriber('/diagnostics', DiagnosticArray, self.diagnostics_callback, queue_size=1)
    
    def play_sound(self, sound_file):
        sound_path = os.path.join(self.sounds_dir, f"{sound_file}")
        if os.path.exists(sound_path):
            data, fs = sf.read(sound_path, dtype='float32')
            sd.play(data, fs, blocking=True)
    
    def diagnostics_callback(self, msg):
        # Skip checks if within startup period
        if rospy.Time.now() - self.start_time < self.startup_ignore_duration:
            return
        
        # Array will always have exactly one element
        message = msg.status[0]
        if message.name in self.monitored_components:
            message_freq_slice, message_delay_slice = message.message.split(",")

            def get_status_level(slice):
                if "error" in slice:
                    return DiagnosticStatus.ERROR
                elif "warning" in slice:
                    return DiagnosticStatus.WARN
                else:
                    return DiagnosticStatus.OK

            def get_status_message(status):
                if status == DiagnosticStatus.ERROR:
                    return "error"
                elif status == DiagnosticStatus.WARN:
                    return "warning"
                else:
                    return "ok"

            now = rospy.Time.now()
            comp = message.name
            
            # --- Frequency ---
            prev_freq_status = self.monitored_components[comp]['freq']['status']
            freq_status = get_status_level(message_freq_slice)
            freq_ok_since = self.monitored_components[comp]['freq']['ok_since']
            freq_history = self.monitored_components[comp]['freq']['history']

            # Update history (keep last 3)
            freq_history.append(freq_status)
            if len(freq_history) > 3:
                freq_history.pop(0)

            if freq_status == DiagnosticStatus.OK:
                if prev_freq_status != DiagnosticStatus.OK:
                    # Just became OK, start timer
                    self.monitored_components[comp]['freq']['ok_since'] = now
                elif freq_ok_since != -1 and (now - freq_ok_since) > self.ok_min_duration:
                    # Has been OK for enough time, play sound and reset timer
                    sound_file = f"{comp.lower()}_frequency_ok.wav"
                    self.soundfile_queue.put(sound_file)
                    self.monitored_components[comp]['freq']['ok_since'] = -1  # Prevent repeated sound
            else:
                # If we are in the special case: WARN -> OK (countdown) -> ERROR (before countdown end)
                if (
                    freq_status == DiagnosticStatus.ERROR and
                    freq_ok_since != -1 and
                    freq_history == [DiagnosticStatus.WARN, DiagnosticStatus.OK, DiagnosticStatus.ERROR]
                ):
                    sound_file = f"{comp.lower()}_frequency_error.wav"
                    self.soundfile_queue.put(sound_file)
                # Reset ok_since if not OK
                self.monitored_components[comp]['freq']['ok_since'] = -1
                # Only play warning/error sound on real transitions (not when Warn -> OK countdown -> Warn OR Error -> OK countdown -> Error)
                if freq_status != prev_freq_status and not (
                    freq_status == DiagnosticStatus.WARN and prev_freq_status == DiagnosticStatus.ERROR
                ):
                    # Don't play if we just handled the special case above
                    if not (
                        freq_status == DiagnosticStatus.ERROR and
                        freq_ok_since != -1 and
                        freq_history == [DiagnosticStatus.WARN, DiagnosticStatus.OK, DiagnosticStatus.ERROR]
                    ) and not (
                        freq_ok_since != -1 and
                        prev_freq_status == DiagnosticStatus.OK
                    ):
                        sound_file = f"{comp.lower()}_frequency_{get_status_message(freq_status)}.wav"
                        self.soundfile_queue.put(sound_file)
            self.monitored_components[comp]['freq']['status'] = freq_status
            self.monitored_components[comp]['freq']['history'] = freq_history

            # --- Delay ---
            prev_delay_status = self.monitored_components[comp]['delay']['status']
            delay_status = get_status_level(message_delay_slice)
            delay_ok_since = self.monitored_components[comp]['delay']['ok_since']
            delay_history = self.monitored_components[comp]['delay']['history']

            # Update history (keep last 3)
            delay_history.append(delay_status)
            if len(delay_history) > 3:
                delay_history.pop(0)

            if delay_status == DiagnosticStatus.OK:
                if prev_delay_status != DiagnosticStatus.OK:
                    self.monitored_components[comp]['delay']['ok_since'] = now
                elif delay_ok_since != -1 and (now - delay_ok_since) > self.ok_min_duration:
                    sound_file = f"{comp.lower()}_delay_ok.wav"
                    self.soundfile_queue.put(sound_file)
                    self.monitored_components[comp]['delay']['ok_since'] = -1
            else:
                if (
                    delay_status == DiagnosticStatus.ERROR and
                    delay_ok_since != -1 and
                    delay_history == [DiagnosticStatus.WARN, DiagnosticStatus.OK, DiagnosticStatus.ERROR]
                ):
                    sound_file = f"{comp.lower()}_delay_error.wav"
                    self.soundfile_queue.put(sound_file)
                self.monitored_components[comp]['delay']['ok_since'] = -1
                if delay_status != prev_delay_status and not (
                    delay_status == DiagnosticStatus.WARN and prev_delay_status == DiagnosticStatus.ERROR
                ):
                    if not (
                        delay_status == DiagnosticStatus.ERROR and
                        delay_ok_since != -1 and
                        delay_history == [DiagnosticStatus.WARN, DiagnosticStatus.OK, DiagnosticStatus.ERROR]
                    ) and not (
                        delay_ok_since != -1 and
                        prev_delay_status == DiagnosticStatus.OK
                    ):
                        sound_file = f"{comp.lower()}_delay_{get_status_message(delay_status)}.wav"
                        self.soundfile_queue.put(sound_file)
            self.monitored_components[comp]['delay']['status'] = delay_status
            self.monitored_components[comp]['delay']['history'] = delay_history
            
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