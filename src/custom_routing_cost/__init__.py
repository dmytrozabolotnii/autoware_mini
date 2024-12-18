from .custom_routing_cost import CustomRoutingCost

class BuslaneRoutingCost(CustomRoutingCost):
    def __init__(self, laneChangeCost, minLaneChangeLength=0.0):
        super(BuslaneRoutingCost, self).__init__(laneChangeCost, minLaneChangeLength)
    
    def getCostSucceeding(self, trafficRules, src, dst):
        cost = super().getCostSucceeding(trafficRules, src, dst)
        if ('subtype' in src.attributes and src.attributes['subtype'] == 'bus_lane') or \
            ('subtype' in dst.attributes and dst.attributes['subtype'] == 'bus_lane'):
            # double the cost for bus lanes to not drive into bus stops
            return cost * 2;
        else:
            return cost
