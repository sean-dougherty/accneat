#pragma once

#include <iostream>
#include <cmath>

namespace NEAT {

    class NodeLocation {
    public:
        short x;
        short y;

        NodeLocation(short x_ = SHRT_MIN, short y_ = SHRT_MIN)
            : x(x_)
            , y(y_) {
        }

        double dist(const NodeLocation &other) const {
            double dx = x - other.x;
            double dy = y - other.y;
            return std::sqrt(dx*dx + dy*dy);
        }

        bool operator==(const NodeLocation &other) const {
            return (x == other.x) && (y == other.y);
        }

        bool operator!=(const NodeLocation &other) const {
            return !(*this == other);
        }

        bool operator<(const NodeLocation &other) const {
            if(x < other.x)
                return true;
            if( (x == other.x) && (y < other.y) )
                return true;

            return false;
        }

        friend std::ostream &operator<<(std::ostream &out, const NodeLocation &loc) {
            return out << loc.x << " " << loc.y;
        }
    };

}
