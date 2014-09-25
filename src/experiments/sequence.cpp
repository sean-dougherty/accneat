#include "experiment.h"

using namespace NEAT;
using namespace std;

static void init_env() {
        const bool DELETE_NODES = true;
        const bool DELETE_LINKS = true;

        if(DELETE_NODES) {
            NEAT::mutate_delete_node_prob = 0.1 * NEAT::mutate_add_node_prob;
        }

        if(DELETE_LINKS) {
            NEAT::mutate_delete_link_prob = 0.1 * NEAT::mutate_add_link_prob;

            NEAT::mutate_toggle_enable_prob = 0.0;
            NEAT::mutate_gene_reenable_prob = 0.0;
        }

        NEAT::compat_threshold = 10.0;
}

const float S = 1.0; // Signal
const float Q = 1.0; // Query
const float _ = 0.0; // Null

const float A = 0.0;
const float B = 1.0;

const real_t weight_seq = 4;
const real_t weight_delay = 25;
const real_t weight_query = 55;

class SequentialInputExperiment : public Experiment {
public:
    SequentialInputExperiment()
        : Experiment("seq-input") {
    }

    virtual void init_env() override {
        ::init_env();
    }

    virtual vector<Test> create_tests() override {
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

    virtual void init_env() override {
        ::init_env();
    }

    virtual vector<Test> create_tests() override {
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
