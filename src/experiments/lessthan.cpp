#include "experiment.h"

using namespace NEAT;
using namespace std;

class LessThanExperiment : public Experiment {
public:
    LessThanExperiment()
        : Experiment("lessthan") {
    }

    virtual vector<Test> create_tests() override {
#define __ 0.0, 0.0
#define _0 0.0, 0.0
#define _1 0.0, 1.0
#define _2 1.0, 0.0
#define _3 1.0, 1.0

#define _ 0.0
#define X 1.0
#define Y 1.0
#define Q 1.0
#define T 1.0
#define F 0.0

        const real_t weight_prob = 0.0;
        const real_t weight_query = 1.0;

        return {
            {{
                    {{X, _, _, _0}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, Y, _, _0}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, _, Q, __}, {F}, weight_query},
            }},
            {{
                    {{X, _, _, _0}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, Y, _, _1}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, _, Q, __}, {T}, weight_query},
            }},
            {{
                    {{X, _, _, _0}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, Y, _, _2}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, _, Q, __}, {T}, weight_query},
            }},
            {{
                    {{X, _, _, _0}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, Y, _, _3}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, _, Q, __}, {T}, weight_query},
            }},
            {{
                    {{X, _, _, _1}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, Y, _, _0}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, _, Q, __}, {F}, weight_query},
            }},
            {{
                    {{X, _, _, _1}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, Y, _, _1}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, _, Q, __}, {F}, weight_query},
            }},
            {{
                    {{X, _, _, _1}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, Y, _, _2}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, _, Q, __}, {T}, weight_query},
            }},
            {{
                    {{X, _, _, _1}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, Y, _, _3}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, _, Q, __}, {T}, weight_query},
            }},
            {{
                    {{X, _, _, _2}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, Y, _, _0}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, _, Q, __}, {F}, weight_query},
            }},
            {{
                    {{X, _, _, _2}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, Y, _, _1}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, _, Q, __}, {F}, weight_query},
            }},
            {{
                    {{X, _, _, _2}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, Y, _, _2}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, _, Q, __}, {F}, weight_query},
            }},
            {{
                    {{X, _, _, _2}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, Y, _, _3}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, _, Q, __}, {T}, weight_query},
            }},
            {{
                    {{X, _, _, _3}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, Y, _, _0}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, _, Q, __}, {F}, weight_query},
            }},
            {{
                    {{X, _, _, _3}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, Y, _, _1}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, _, Q, __}, {F}, weight_query},
            }},
            {{
                    {{X, _, _, _3}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, Y, _, _2}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, _, Q, __}, {F}, weight_query},
            }},
            {{
                    {{X, _, _, _3}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, Y, _, _3}, {_}, weight_prob},
                    {{_, _, _, __}, {_}, weight_prob},
                    {{_, _, Q, __}, {F}, weight_query}
            }}
        };
    }
} lessthan;
