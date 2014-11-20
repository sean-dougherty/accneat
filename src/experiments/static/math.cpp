#include "std.h" // Must be included first. Precompiled header with standard library includes.
#include "staticexperiment.h"

using namespace NEAT;
using namespace std;

static struct MathInit {
    MathInit() {
        create_static_experiment("add-1bit", [] () {
                const real_t weight = 1.0;

                vector<Test> tests = {
                    {{
                            {{0.0, 0.0}, {0.0, 0.0}, weight},
                    }},
                    {{
                            {{0.0, 1.0}, {0.0, 1.0}, weight},
                    }},
                    {{
                            {{1.0, 0.0}, {0.0, 1.0}, weight},
                    }},
                    {{
                            {{1.0, 1.0}, {1.0, 0.0}, weight}
                    }}
                };

                return tests;
            });

        create_static_experiment("add-2bit", [] () {
                const real_t weight = 1.0;
#define _0 0.0, 0.0, 0.0
#define _1 0.0, 0.0, 1.0
#define _2 0.0, 1.0, 0.0
#define _3 0.0, 1.0, 1.0
#define _4 1.0, 0.0, 0.0
#define _5 1.0, 0.0, 1.0
#define _6 1.0, 1.0, 0.0

                vector<Test> tests = {
                    {{
                            {{_0, _0}, {_0}, weight},
                    }},
                    {{
                            {{_0, _1}, {_1}, weight},
                    }},
                    {{
                            {{_0, _2}, {_2}, weight},
                    }},
                    {{
                            {{_0, _3}, {_3}, weight},
                    }},
                    {{
                            {{_1, _0}, {_1}, weight},
                    }},
                    {{
                            {{_1, _1}, {_2}, weight},
                    }},
                    {{
                            {{_1, _2}, {_3}, weight},
                    }},
                    {{
                            {{_1, _3}, {_4}, weight},
                    }},
                    {{
                            {{_2, _0}, {_2}, weight},
                    }},
                    {{
                            {{_2, _1}, {_3}, weight},
                    }},
                    {{
                            {{_2, _2}, {_4}, weight},
                    }},
                    {{
                            {{_2, _3}, {_5}, weight},
                    }},
                    {{
                            {{_3, _0}, {_3}, weight},
                    }},
                    {{
                            {{_3, _1}, {_4}, weight},
                    }},
                    {{
                            {{_3, _2}, {_5}, weight},
                    }},
                    {{
                            {{_3, _3}, {_6}, weight},
                    }},
                };

                return tests;
            });
    }
} init;
