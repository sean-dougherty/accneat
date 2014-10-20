#include "std.h" // Must be included first. Precompiled header with standard library includes.
#include "batteryexperiment.h"
#include <assert.h>

using namespace NEAT;
using namespace std;

static Test create_test(string sym_order);

class LessThanExperiment : public BatteryExperiment {
public:
    LessThanExperiment()
        : BatteryExperiment("lessthan") {
    }

    virtual vector<Test> create_tests() override {
        return {
            create_test("abc")
        };
    }
} lessthan;

static Test create_test(string sym_order) {
    static map<char, array<real_t, 2>> sym_encoding {
        {'a', {0.0, 1.0}},
        {'b', {1.0, 0.0}},
        {'c', {1.0, 1.0}}
    };
    const real_t weight_query = 1.0;

    assert(sym_order.size() == 3);

    map<char, size_t> sym_val;
    for(char sym: sym_order) {
        assert(sym >= 'a' && sym <= 'c');
        size_t val = sym_val.size();
        sym_val[sym] = val;
    }

    vector<Step> steps;
    for(char x_sym = 'a'; x_sym <= 'c'; x_sym++) {
        for(char y_sym = 'a'; y_sym <= 'c'; y_sym++) {
            if(x_sym == y_sym) {
                continue;
            }

            auto x_enc = sym_encoding[x_sym];
            size_t x_val = sym_val[x_sym];
            auto y_enc = sym_encoding[y_sym];
            size_t y_val = sym_val[y_sym];

            steps.push_back({
                {x_enc[0], x_enc[1], y_enc[0], y_enc[1]},
                {x_val < y_val ? real_t(1.0) : real_t(0.0)},
                weight_query
            });
        }
    }

    return {steps};
}
