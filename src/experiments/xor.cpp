#include "experiment.h"

using namespace NEAT;
using namespace std;

class XorExperiment : public Experiment {
public:
    XorExperiment()
        : Experiment("xor") {
    }

    virtual void init_env() override {
        env->compat_threshold = 10.0;
    }

    virtual vector<Test> create_tests() override {
        const real_t T = 1.0;
        const real_t F = 0.0;
        const real_t weight = 1.0;

        return {
            {{
                    {{F, F}, {F}, weight},
            }},
            {{
                    {{F, T}, {T}, weight},
            }},
            {{
                    {{T, F}, {T}, weight},
            }},
            {{
                    {{T, T}, {F}, weight}
            }}
        };
    }
} xor_experiment;
