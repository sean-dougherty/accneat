#pragma once

#include <algorithm>
#include <memory>

namespace NEAT {

#define warn(msg) {std::cout << __FILE__ << ":" << __LINE__ << ": " << msg << std::endl;}
#define trap(msg) {std::cout << __FILE__ << ":" << __LINE__ << ": " << msg << std::endl; abort();}
#define impl() {std::cout << __FILE__ << ":" << __LINE__ << ": IMPLEMENT!" << std::endl; abort();}
#define panic() {std::cout << __FILE__ << ":" << __LINE__ << ": PANIC!" << std::endl; abort();}

    template<typename Container, typename Predicate>
    void erase_if(Container &cont, Predicate predicate) {
        auto iterator = std::remove_if(cont.begin(), cont.end(), predicate);
        cont.erase(iterator, cont.end());
    }

    template<typename T, typename... Args>
    std::unique_ptr<T> make_unique(Args&&... args)
    {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }
}
