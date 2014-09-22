#pragma once

#include <cstdlib>
#include <vector>

namespace NEAT {
    class Timer {
        static std::vector<Timer *> timers;

        const char *_name;
        size_t _n = 0;
        double _total = 0.0;
        double _min = 0.0;
        double _max = 0.0;
        double _start = 0.0;
        double _recent = 0.0;
    public:
        Timer(const char *name);
        ~Timer();
    
        void start();
        void stop();

        static void report();
    };
}
