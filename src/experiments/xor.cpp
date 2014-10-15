#include "std.h" // Must be included first. Precompiled header with standard library includes.
#include "experiment.h"

using namespace NEAT;
using namespace std;

class XorExperiment : public Experiment {
public:
    XorExperiment()
        : Experiment("xor") {
    }

    virtual vector<Test> create_tests() override {
        const real_t T = 1.0;
        const real_t F = 0.0;
        const real_t weight = 1.0;

        return {
            {{
                    {{T, F}, {T}, weight},
            }},
            {{
                    {{F, F}, {F}, weight},
            }},
            {{
                    {{F, T}, {T}, weight},
            }},
            {{
                    {{T, T}, {F}, weight}
            }}
        };
    }
} xor_experiment;
