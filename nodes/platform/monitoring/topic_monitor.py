#!/usr/bin/env python3

import rospy
import csv
from rospy.msg import TopicStatistics, DiagnosticStatus, DiagnosticArray, KeyValue

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
        if topic in self.monitoring_config:
            
            status = DiagnosticStatus()
            status.name = self.monitoring_config[topic]['component']
            # Check frequency
            income_freq = msg.period_mean
            if income_freq < self.monitoring_config[topic]['error_freq']:
                status = DiagnosticStatus.ERROR
            
            # TODO
            
            # Check delay
            
            pass
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('topic_monitor')
    monitor = TopicMonitor()
    monitor.run()