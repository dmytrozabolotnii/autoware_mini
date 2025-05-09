#!/usr/bin/env python3

import rospy
import csv
from rosgraph_msgs.msg import TopicStatistics
from diagnostic_msgs.msg import DiagnosticStatus, DiagnosticArray
from rospy.msg import AnyMsg

class TopicMonitor:
    def __init__(self):
        
        # Parameters
        self.monitoring_conf_path = rospy.get_param('~monitoring_conf_path')
        self.monitoring_config = self.load_monitoring_config()
        
        # Other initializations
        self.avg_values = {
        } # Exponential moving average values for frequency and delay
        
        # Publishers
        self.diagnostics_pub = rospy.Publisher('/diagnostics', DiagnosticArray, queue_size=5)
        
        # Subscribers
        rospy.Subscriber('/statistics', TopicStatistics, self.topic_statistics_callback, queue_size=1)
                
        # Dummy subscribers (otherwise topics may not be monitored)
        self.dummy_subs = []
        for topic in self.monitoring_config.keys():
            dummy_sub = rospy.Subscriber(topic, AnyMsg, self.dummy_callback, queue_size=1)
            self.dummy_subs.append(dummy_sub)
    
    def dummy_callback(self, msg):
        # Dummy callback to keep the subscriber alive
        pass
    
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
            avgs_present = self.avg_values.get(topic) is not None
            if not avgs_present:
                self.avg_values[topic] = {
                    'freq': 0.0,
                    'delay': 0.0,
                }
            
            # Check frequency
            income_freq = 1.0 / msg.period_mean.to_sec() if msg.period_mean.to_sec() > 0 else 0
            if avgs_present:
                self.avg_values[topic]['freq'] = 0.2 * self.avg_values[topic]['freq'] + 0.8 * income_freq
            else:
                self.avg_values[topic]['freq'] = income_freq 
            avg_freq = self.avg_values[topic]['freq']
            
            if avg_freq < self.monitoring_config[topic]['error_freq']:
                status.level = DiagnosticStatus.ERROR
                status.message = f"{status.name} frequency error: {avg_freq:.3f} Hz"
            elif avg_freq < self.monitoring_config[topic]['warning_freq']:
                status.level = DiagnosticStatus.WARN
                status.message = f"{status.name} frequency warning: {avg_freq:.3f} Hz"
            else:
                status.level = DiagnosticStatus.OK
                status.message = f"{status.name} frequency nominal: {avg_freq:.3f} Hz"
                        
            # Check delay
            delay = msg.stamp_age_mean.to_sec()
            if avgs_present:
                self.avg_values[topic]['delay'] = 0.2 * self.avg_values[topic]['delay'] + 0.8 * delay
            else:
                self.avg_values[topic]['delay'] = delay
            avg_delay = self.avg_values[topic]['delay'] 
            
            if avg_delay > self.monitoring_config[topic]['error_delay']:
                status.level = max(status.level, DiagnosticStatus.ERROR)  # Escalate to ERROR if necessary
                status.message += f", {status.name} delay error: {avg_delay:.3f} s"
            elif avg_delay > self.monitoring_config[topic]['warning_delay']:
                status.level = max(status.level, DiagnosticStatus.WARN)  # Escalate to WARN if necessary
                status.message += f", {status.name} delay warning: {avg_delay:.3f} s"
            else:
                status.message += f", {status.name} delay nominal: {avg_delay:.3f} s"            
                
            diagnostics_array.status.append(status)
            self.diagnostics_pub.publish(diagnostics_array)
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('topic_monitor')
    monitor = TopicMonitor()
    monitor.run()