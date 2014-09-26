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

class XorExperiment : public Experiment {
public:
    XorExperiment()
        : Experiment("xor") {
    }

    virtual void init_env() override {
        ::init_env();
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
