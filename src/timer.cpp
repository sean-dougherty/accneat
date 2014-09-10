#include "timer.h"

#include <assert.h>
#include <sys/time.h>

#include <algorithm>
#include <iostream>

using namespace std;

vector<Timer *> Timer::timers;

static double seconds() {
    struct timeval tv;
    gettimeofday( &tv, NULL );

    return (double)tv.tv_sec + (double)tv.tv_usec/1000000.0;
}

Timer::Timer(const char *name) : _name(name) {
    timers.push_back(this);
}

Timer::~Timer() {
    timers.erase(find(timers.begin(), timers.end(), this));
}

void Timer::start() {
    assert(_start == 0.0);

    _start = seconds();
}

void Timer::stop() {
    assert(_start != 0.0);

    double t = seconds() - _start;
    _start = 0.0;

    if(_n == 0) {
        _min = _max = t;
    } else {
        _min = min(_min, t);
        _max = max(_max, t);
    }
    _total += t;
    _n++;
}

void Timer::report() {
    for(Timer *t: timers) {
        cout << t->_name << ": n=" << t->_n << ", total=" << t->_total << ", mean=" << (t->_total / t->_n) << ", min=" << t->_min << ", max=" << t->_max << endl;
    }
}
