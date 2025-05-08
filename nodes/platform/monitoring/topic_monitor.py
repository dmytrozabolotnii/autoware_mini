#!/usr/bin/env python3

import rospy
import csv
from rosgraph_msgs.msg import TopicStatistics
from diagnostic_msgs.msg import DiagnosticStatus, DiagnosticArray

class TopicMonitor:
    def __init__(self):
        
        # Parameters
        self.monitoring_conf_path = rospy.get_param('~monitoring_conf_path')
        self.monitoring_config = self.load_monitoring_config()
        
        # Publishers
        self.diagnostics_pub = rospy.Publisher('/diagnostics', DiagnosticArray, queue_size=5)
        
        # Subscribers
        rospy.Subscriber('/statistics', TopicStatistics, self.topic_statistics_callback, queue_size=1)
    
    def load_monitoring_config(self):
        config = {}
        with open(self.monitoring_conf_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                topic = row['topic']
                config[topic] = {
                    'component': row['component'],
                    'nominal_freq': float(row['nominal_freq']),
                    'warning_freq': float(row['warning_freq']),
                    'error_freq': float(row['error_freq']),
                    'nominal_delay': float(row['nominal_delay']),
                    'warning_delay': float(row['warning_delay']),
                    'error_delay': float(row['error_delay']),
                }
        return config
    
    def topic_statistics_callback(self, msg):
        topic = msg.topic
        diagnostics_array = DiagnosticArray()
        diagnostics_array.header.stamp = rospy.Time.now()
        if topic in self.monitoring_config:
            
            status = DiagnosticStatus()
            status.name = self.monitoring_config[topic]['component']
            # Check frequency
            income_freq = 1.0 / msg.period_mean.to_sec() if msg.period_mean.to_sec() > 0 else 0
            if income_freq < self.monitoring_config[topic]['error_freq']:
                status.level = DiagnosticStatus.ERROR
                status.message = f"{status.name} frequency error: {income_freq:.2f} Hz"
            elif income_freq < self.monitoring_config[topic]['warning_freq']:
                status.level = DiagnosticStatus.WARN
                status.message = f"{status.name} frequency warning: {income_freq:.2f} Hz"
            else:
                status.level = DiagnosticStatus.OK
                status.message = f"{status.name} frequency nominal: {income_freq:.2f} Hz"
                        
            # Check delay
            delay = msg.stamp_age_mean.to_sec()
            if delay > self.monitoring_config[topic]['error_delay']:
                status.level = max(status.level, DiagnosticStatus.ERROR)  # Escalate to ERROR if necessary
                status.message += f", {status.name} delay error: {delay:.2f} s"
            elif delay > self.monitoring_config[topic]['warning_delay']:
                status.level = max(status.level, DiagnosticStatus.WARN)  # Escalate to WARN if necessary
                status.message += f", {status.name} delay warning: {delay:.2f} s"
            else:
                status.message += f", {status.name} delay nominal: {delay:.2f} s"            
                
            diagnostics_array.status.append(status)
            self.diagnostics_pub.publish(diagnostics_array)
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('topic_monitor')
    monitor = TopicMonitor()
    monitor.run()