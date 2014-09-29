#include "experiment.h"

using namespace NEAT;
using namespace std;

class SequentialInputExperiment : public Experiment {
public:
    SequentialInputExperiment()
        : Experiment("seq-input") {
    }

    virtual vector<Test> create_tests() override {
        const float A = 0.0;
        const float B = 1.0;

        const float S = 1.0; // Signal
        const float Q = 1.0; // Query
        const float _ = 0.0; // Null

        const real_t weight_seq = 4;
        const real_t weight_delay = 25;
        const real_t weight_query = 55;

        return {
            {{
                    {{S, _, _, A, _}, {_, _, _}, weight_seq},
                    {{_, _, _, _, _}, {_, _, _}, weight_seq},
                    {{S, _, _, A, _}, {_, _, _}, weight_seq},
                    {{_, _, _, _, _}, {_, _, _}, weight_seq},
                    {{S, _, _, A, _}, {_, _, _}, weight_seq},
                    {{_, _, _, _, _}, {_, _, _}, weight_delay},
                    {{_, _, Q, _, _}, {A, A, A}, weight_query}
            }},
            {{
                    {{S, _, _, A, _}, {_, _, _}, weight_seq},
                    {{_, _, _, _, _}, {_, _, _}, weight_seq},
                    {{S, _, _, A, _}, {_, _, _}, weight_seq},
                    {{_, _, _, _, _}, {_, _, _}, weight_seq},
                    {{S, _, _, B, _}, {_, _, _}, weight_seq},
                    {{_, _, _, _, _}, {_, _, _}, weight_delay},
                    {{_, _, Q, _, _}, {A, A, B}, weight_query}
            }},
            {{
                    {{S, _, _, A, _}, {_, _, _}, weight_seq},
                    {{_, _, _, _, _}, {_, _, _}, weight_seq},
                    {{S, _, _, B, _}, {_, _, _}, weight_seq},
                    {{_, _, _, _, _}, {_, _, _}, weight_seq},
                    {{S, _, _, A, _}, {_, _, _}, weight_seq},
                    {{_, _, _, _, _}, {_, _, _}, weight_delay},
                    {{_, _, Q, _, _}, {A, B, A}, weight_query}
            }},
            {{
                    {{S, _, _, A, _}, {_, _, _}, weight_seq},
                    {{_, _, _, _, _}, {_, _, _}, weight_seq},
                    {{S, _, _, B, _}, {_, _, _}, weight_seq},
                    {{_, _, _, _, _}, {_, _, _}, weight_seq},
                    {{S, _, _, B, _}, {_, _, _}, weight_seq},
                    {{_, _, _, _, _}, {_, _, _}, weight_delay},
                    {{_, _, Q, _, _}, {A, B, B}, weight_query}
            }},
            {{
                    {{S, _, _, B, _}, {_, _, _}, weight_seq},
                    {{_, _, _, _, _}, {_, _, _}, weight_seq},
                    {{S, _, _, A, _}, {_, _, _}, weight_seq},
                    {{_, _, _, _, _}, {_, _, _}, weight_seq},
                    {{S, _, _, A, _}, {_, _, _}, weight_seq},
                    {{_, _, _, _, _}, {_, _, _}, weight_delay},
                    {{_, _, Q, _, _}, {B, A, A}, weight_query}
            }},
            {{
                    {{S, _, _, B, _}, {_, _, _}, weight_seq},
                    {{_, _, _, _, _}, {_, _, _}, weight_seq},
                    {{S, _, _, A, _}, {_, _, _}, weight_seq},
                    {{_, _, _, _, _}, {_, _, _}, weight_seq},
                    {{S, _, _, B, _}, {_, _, _}, weight_seq},
                    {{_, _, _, _, _}, {_, _, _}, weight_delay},
                    {{_, _, Q, _, _}, {B, A, B}, weight_query}
            }},
            {{
                    {{S, _, _, B, _}, {_, _, _}, weight_seq},
                    {{_, _, _, _, _}, {_, _, _}, weight_seq},
                    {{S, _, _, B, _}, {_, _, _}, weight_seq},
                    {{_, _, _, _, _}, {_, _, _}, weight_seq},
                    {{S, _, _, A, _}, {_, _, _}, weight_seq},
                    {{_, _, _, _, _}, {_, _, _}, weight_delay},
                    {{_, _, Q, _, _}, {B, B, A}, weight_query}
            }},
            {{
                    {{S, _, _, B, _}, {_, _, _}, weight_seq},
                    {{_, _, _, _, _}, {_, _, _}, weight_seq},
                    {{S, _, _, B, _}, {_, _, _}, weight_seq},
                    {{_, _, _, _, _}, {_, _, _}, weight_seq},
                    {{S, _, _, B, _}, {_, _, _}, weight_seq},
                    {{_, _, _, _, _}, {_, _, _}, weight_delay},
                    {{_, _, Q, _, _}, {B, B, B}, weight_query}
            }}
        };
    }
} seq_input;

class SequentialOutputExperiment : public Experiment {
public:
    SequentialOutputExperiment()
        : Experiment("seq-output") {
    }

    virtual vector<Test> create_tests() override {
        const float A = 0.0;
        const float B = 1.0;

        const float S = 1.0; // Signal
        const float Q = 1.0; // Query
        const float _ = 0.0; // Null

        const real_t weight_seq = 4;
        const real_t weight_delay = 25;
        const real_t weight_query = 55;

        return {
            {{
                    {{S, A, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{S, A, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{S, A, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, _}, {_}, weight_delay},
                    {{_, _, Q, _, _}, {A}, weight_query},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{_, _, _, Q, _}, {A}, weight_query},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, Q}, {A}, weight_query}
            }},
            {{
                    {{S, A, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{S, A, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{S, B, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, _}, {_}, weight_delay},
                    {{_, _, Q, _, _}, {A}, weight_query},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{_, _, _, Q, _}, {A}, weight_query},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, Q}, {B}, weight_query}
            }},
            {{
                    {{S, A, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{S, B, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{S, A, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, _}, {_}, weight_delay},
                    {{_, _, Q, _, _}, {A}, weight_query},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{_, _, _, Q, _}, {B}, weight_query},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, Q}, {A}, weight_query}
            }},
            {{
                    {{S, A, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{S, B, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{S, B, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, _}, {_}, weight_delay},
                    {{_, _, Q, _, _}, {A}, weight_query},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{_, _, _, Q, _}, {B}, weight_query},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, Q}, {B}, weight_query}
            }},
            {{
                    {{S, B, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{S, A, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{S, A, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, _}, {_}, weight_delay},
                    {{_, _, Q, _, _}, {B}, weight_query},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{_, _, _, Q, _}, {A}, weight_query},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, Q}, {A}, weight_query}
            }},
            {{
                    {{S, B, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{S, A, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{S, B, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, _}, {_}, weight_delay},
                    {{_, _, Q, _, _}, {B}, weight_query},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{_, _, _, Q, _}, {A}, weight_query},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, Q}, {B}, weight_query}
            }},
            {{
                    {{S, B, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{S, B, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{S, A, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, _}, {_}, weight_delay},
                    {{_, _, Q, _, _}, {B}, weight_query},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{_, _, _, Q, _}, {B}, weight_query},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, Q}, {A}, weight_query}
            }},
            {{
                    {{S, B, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{S, B, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{S, B, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, _}, {_}, weight_delay},
                    {{_, _, Q, _, _}, {B}, weight_query},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{_, _, _, Q, _}, {B}, weight_query},
                    {{_, _, _, _, _}, {_}, weight_seq},
                    {{_, _, _, _, Q}, {B}, weight_query}
            }}
        };
    }
} seq_output;

class SequentialAbcExperiment : public Experiment {
public:
    SequentialAbcExperiment()
        : Experiment("seq-abc") {
    }

    virtual vector<Test> create_tests() override {
        const float A = 0.0;
        const float B = 0.5;
        const float C = 1.0;

        const float S = 1.0; // Signal
        const float Q = 1.0; // Query
        const float _ = 0.0; // Null

        const real_t weight_seq = 4;
        const real_t weight_delay = 25;
        const real_t weight_query = 55;

        return {
            {{
                    {{S, _, A}, {_, _}, weight_seq},
                    {{_, _, _}, {_, _}, weight_seq},
                    {{S, _, A}, {_, _}, weight_seq},
                    {{_, _, _}, {_, _}, weight_delay},
                    {{_, Q, _}, {A, A}, weight_query}
            }},
            {{
                    {{S, _, A}, {_, _}, weight_seq},
                    {{_, _, _}, {_, _}, weight_seq},
                    {{S, _, B}, {_, _}, weight_seq},
                    {{_, _, _}, {_, _}, weight_delay},
                    {{_, Q, _}, {A, B}, weight_query}
            }},
            {{
                    {{S, _, A}, {_, _}, weight_seq},
                    {{_, _, _}, {_, _}, weight_seq},
                    {{S, _, C}, {_, _}, weight_seq},
                    {{_, _, _}, {_, _}, weight_delay},
                    {{_, Q, _}, {A, C}, weight_query}
            }},
            {{
                    {{S, _, B}, {_, _}, weight_seq},
                    {{_, _, _}, {_, _}, weight_seq},
                    {{S, _, A}, {_, _}, weight_seq},
                    {{_, _, _}, {_, _}, weight_delay},
                    {{_, Q, _}, {B, A}, weight_query}
            }},
            {{
                    {{S, _, B}, {_, _}, weight_seq},
                    {{_, _, _}, {_, _}, weight_seq},
                    {{S, _, B}, {_, _}, weight_seq},
                    {{_, _, _}, {_, _}, weight_delay},
                    {{_, Q, _}, {B, B}, weight_query}
            }},
            {{
                    {{S, _, B}, {_, _}, weight_seq},
                    {{_, _, _}, {_, _}, weight_seq},
                    {{S, _, C}, {_, _}, weight_seq},
                    {{_, _, _}, {_, _}, weight_delay},
                    {{_, Q, _}, {B, C}, weight_query}
            }},
            {{
                    {{S, _, C}, {_, _}, weight_seq},
                    {{_, _, _}, {_, _}, weight_seq},
                    {{S, _, A}, {_, _}, weight_seq},
                    {{_, _, _}, {_, _}, weight_delay},
                    {{_, Q, _}, {C, A}, weight_query}
            }},
            {{
                    {{S, _, C}, {_, _}, weight_seq},
                    {{_, _, _}, {_, _}, weight_seq},
                    {{S, _, B}, {_, _}, weight_seq},
                    {{_, _, _}, {_, _}, weight_delay},
                    {{_, Q, _}, {C, B}, weight_query}
            }},
            {{
                    {{S, _, C}, {_, _}, weight_seq},
                    {{_, _, _}, {_, _}, weight_seq},
                    {{S, _, C}, {_, _}, weight_seq},
                    {{_, _, _}, {_, _}, weight_delay},
                    {{_, Q, _}, {C, C}, weight_query}
            }},
        };
    }
} seq_abc;

class Sequential2bitExperiment : public Experiment {
public:
    Sequential2bitExperiment()
        : Experiment("seq-2bit") {
    }

    virtual vector<Test> create_tests() override {
#define _A 0.0, 0.0
#define _B 0.0, 1.0
#define _C 1.0, 0.0
#define _D 1.0, 1.0
#define __ 0.0, 0.0

        const float S = 1.0; // Signal
        const float Q = 1.0; // Query
        const float _ = 0.0; // Null

        const real_t weight_seq = 4;
        const real_t weight_delay = 25;
        const real_t weight_query = 55;

        return {
            {{
                    {{S, _, _A}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_seq},
                    {{S, _, _A}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_delay},
                    {{_, Q, __}, {_A, _A}, weight_query}
            }},                        
            {{                         
                    {{S, _, _A}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_seq},
                    {{S, _, _B}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_delay},
                    {{_, Q, __}, {_A, _B}, weight_query}
            }},                        
            {{                         
                    {{S, _, _A}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_seq},
                    {{S, _, _C}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_delay},
                    {{_, Q, __}, {_A, _C}, weight_query}
            }},                        
            {{                         
                    {{S, _, _A}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_seq},
                    {{S, _, _D}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_delay},
                    {{_, Q, __}, {_A, _D}, weight_query}
            }},                        
            {{                         
                    {{S, _, _B}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_seq},
                    {{S, _, _A}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_delay},
                    {{_, Q, __}, {_B, _A}, weight_query}
            }},                        
            {{                         
                    {{S, _, _B}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_seq},
                    {{S, _, _B}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_delay},
                    {{_, Q, __}, {_B, _B}, weight_query}
            }},                        
            {{                         
                    {{S, _, _B}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_seq},
                    {{S, _, _C}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_delay},
                    {{_, Q, __}, {_B, _C}, weight_query}
            }},                        
            {{                         
                    {{S, _, _B}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_seq},
                    {{S, _, _D}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_delay},
                    {{_, Q, __}, {_B, _D}, weight_query}
            }},                        
            {{                         
                    {{S, _, _C}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_seq},
                    {{S, _, _A}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_delay},
                    {{_, Q, __}, {_C, _A}, weight_query}
            }},                        
            {{                         
                    {{S, _, _C}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_seq},
                    {{S, _, _B}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_delay},
                    {{_, Q, __}, {_C, _B}, weight_query}
            }},                        
            {{                         
                    {{S, _, _C}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_seq},
                    {{S, _, _C}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_delay},
                    {{_, Q, __}, {_C, _C}, weight_query}
            }},                        
            {{                         
                    {{S, _, _C}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_seq},
                    {{S, _, _D}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_delay},
                    {{_, Q, __}, {_C, _D}, weight_query}
            }},                        
            {{                         
                    {{S, _, _D}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_seq},
                    {{S, _, _A}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_delay},
                    {{_, Q, __}, {_D, _A}, weight_query}
            }},                        
            {{                         
                    {{S, _, _D}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_seq},
                    {{S, _, _B}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_delay},
                    {{_, Q, __}, {_D, _B}, weight_query}
            }},                        
            {{                         
                    {{S, _, _D}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_seq},
                    {{S, _, _C}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_delay},
                    {{_, Q, __}, {_D, _C}, weight_query}
            }},                        
            {{                         
                    {{S, _, _D}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_seq},
                    {{S, _, _D}, {__, __}, weight_seq},
                    {{_, _, __}, {__, __}, weight_delay},
                    {{_, Q, __}, {_D, _D}, weight_query}
            }}
        };

#undef _A
#undef _B
#undef _C
#undef _D
#undef __
    }
} seq_2bit;
