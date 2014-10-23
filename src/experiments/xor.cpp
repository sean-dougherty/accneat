#include "std.h" // Must be included first. Precompiled header with standard library includes.
#include "staticexperiment.h"

using namespace NEAT;
using namespace std;

static struct Init {
    Init() {
        create_static_experiment("xor", [] () {
                const real_t T = 1.0;
                const real_t F = 0.0;
                const real_t weight = 1.0;

                vector<Test> tests = {
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

                return tests;
            });
    }
} init;
