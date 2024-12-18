#include <boost/python.hpp>
#include <boost/python/wrapper.hpp>
#include <iostream>

#include <lanelet2_core/primitives/Lanelet.h>
#include <lanelet2_traffic_rules/TrafficRules.h>
#include <lanelet2_routing/RoutingCost.h>

namespace bp = boost::python;

class CustomRoutingCost : public lanelet::routing::RoutingCostDistance, public bp::wrapper<lanelet::routing::RoutingCostDistance> {
public:
    CustomRoutingCost(double laneChangeCost, double minLaneChangeLength=0.0)
        : lanelet::routing::RoutingCostDistance(laneChangeCost, minLaneChangeLength) {}

    virtual double getCostSucceeding(const lanelet::traffic_rules::TrafficRules& tr,
                                     const lanelet::ConstLaneletOrArea& src,
                                     const lanelet::ConstLaneletOrArea& dst) const override {
        if (bp::override f = this->get_override("getCostSucceeding")) {
            //std::cout << "Calling custom getCostSucceeding" << std::endl;
            return f(boost::ref(tr), src, dst);
        }
        //std::cout << "Returning parent getCostSucceeding" << std::endl;
        return lanelet::routing::RoutingCostDistance::getCostSucceeding(tr, src, dst);
    }

    double default_getCostSucceeding(const lanelet::traffic_rules::TrafficRules& tr,
                                     const lanelet::ConstLaneletOrArea& src,
                                     const lanelet::ConstLaneletOrArea& dst) const {
        //std::cout << "Calling default getCostSucceeding" << std::endl;
        return lanelet::routing::RoutingCostDistance::getCostSucceeding(tr, src, dst);
    }

    virtual double getCostLaneChange(const lanelet::traffic_rules::TrafficRules& tr,
                                     const lanelet::ConstLanelets& src,
                                     const lanelet::ConstLanelets& dst) const noexcept override {
        if (bp::override f = this->get_override("getCostLaneChange")) {
            //std::cout << "Calling custom getCostLaneChange" << std::endl;
            return f(boost::ref(tr), src, dst);
        }
        //std::cout << "Returning parent getCostLaneChange" << std::endl;
        return lanelet::routing::RoutingCostDistance::getCostLaneChange(tr, src, dst);
    }

    double default_getCostLaneChange(const lanelet::traffic_rules::TrafficRules& tr,
                                     const lanelet::ConstLanelets& src,
                                     const lanelet::ConstLanelets& dst) const {
        //std::cout << "Calling default getCostLaneChange" << std::endl;
        return lanelet::routing::RoutingCostDistance::getCostLaneChange(tr, src, dst);
    }
};

BOOST_PYTHON_MODULE(custom_routing_cost) {
    bp::class_<CustomRoutingCost, std::shared_ptr<CustomRoutingCost>, boost::noncopyable>(
            "CustomRoutingCost",
            bp::init<double, double>((bp::arg("laneChangeCost"), bp::arg("minLaneChangeLength")=0.0))
        )
        .def("getCostSucceeding", &CustomRoutingCost::default_getCostSucceeding)
        .def("getCostLaneChange", &CustomRoutingCost::default_getCostLaneChange)
    ;
}
