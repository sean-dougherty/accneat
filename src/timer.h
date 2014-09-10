#pragma once

#include <cstdlib>
#include <vector>

class Timer {
    static std::vector<Timer *> timers;

    const char *_name;
    size_t _n = 0;
    double _total = 0.0;
    double _min;
    double _max;
    double _start = 0.0;
public:
    Timer(const char *name);
    ~Timer();
    
    void start();
    void stop();

    static void report();
};
