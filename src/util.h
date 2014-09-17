#pragma once

#include <algorithm>

namespace NEAT {

#define warn(msg) {std::cout << __FILE__ << ":" << __LINE__ << ": " << msg << std::endl;}
#define trap(msg) {std::cout << __FILE__ << ":" << __LINE__ << ": " << msg << std::endl; abort();}

    template<typename Container, typename Predicate>
    void erase_if(Container &cont, Predicate predicate) {
       
        auto iterator = std::remove_if(cont.begin(), cont.end(), predicate);
        cont.erase(iterator, cont.end());
    }
}
