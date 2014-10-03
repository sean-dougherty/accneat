#include "experiment.h"

#include "util.h"
#include <assert.h>
#include <cmath>

using namespace NEAT;
using namespace std;

static vector<Test> create_parallel_output_tests(string syms,
                                                 vector<string> &sequences,
                                                 Test::Type test_type) {
    const real_t weight_seq = 5;
    const real_t weight_query = 50;

    Step::ErrType err_type;
    switch(test_type) {
    case Test::Training:
        err_type = Step::Err_Delta;
        break;
    case Test::Fittest:
        err_type = Step::Err_Binary;
        break;
    }

    assert(syms.size() > 1);
    assert(sequences.size() > 1);
    for(size_t i = 1; i < sequences.size(); i++) {
        assert(sequences[0].size() == sequences[i].size());
    }

    size_t sequence_len = sequences[0].size();
    size_t nsyms = syms.size();
    size_t nbits = ceil(log2(nsyms));

    map<char, vector<real_t>> sym_encoding;
    //Create binary encoding for each symbol
    for(size_t i = 0; i < syms.size(); i++) {
        char sym = syms[i];
        assert(sym_encoding.find(sym) == sym_encoding.end());
        vector<real_t> &encoding = sym_encoding[sym];
        for(size_t bit = nbits; bit > 0; bit--) {
            if(i & (1 << (bit-1))) {
                encoding.push_back(1.0);
            } else {
                encoding.push_back(0.0);
            }
        }
    }

    const real_t _ = 0.0;
    const real_t X = 1.0;

    vector<Test> tests;
    for(string &sequence: sequences) {
        vector<Step> steps;

        //Present sequence
        for(char sym: sequence) {
            //Create step in which symbol is presented
            {
                vector<real_t> input;
                append(input, X); // Symbol being provided in this step
                append(input, _); // Not querying
                append(input, sym_encoding[sym]);

                vector<real_t> output;
                append(output, _, sequence_len * nbits); // Empty output

                steps.emplace_back(input, output, weight_seq, err_type);
            }
            
            //Create silence
            {
                vector<real_t> input;
                append(input, _); // No symbol this step
                append(input, _); // Not querying
                append(input, _, nbits); // Empty symbol

                vector<real_t> output;
                append(output, _, sequence_len * nbits); // Empty output

                steps.emplace_back(input, output, weight_seq, err_type);
            }
        }

        // Query
        {
            vector<real_t> input;
            append(input, _); // No symbol
            append(input, X); // Querying
            append(input, _, nbits); // Empty symbol
            
            vector<real_t> output;
            for(char sym: sequence) {
                append(output, sym_encoding[sym]);
            }

            steps.emplace_back(input, output, weight_query, err_type);
        }

        tests.emplace_back(sequence, steps, test_type);
    }

    return tests;
}

struct FooExperiment : public Experiment {
    FooExperiment() : Experiment("foo") {
    }

    virtual vector<Test> create_tests() override {
        string syms = "ab";
        vector<string> all_sequences = permute_repeat(syms, 2);

        return create_parallel_output_tests(syms, all_sequences, Test::Training);
/*
        vector<string> training;
        vector<string> fittest;
        rng_t rng;

        for(string seq: all_sequences) {
            if(rng.under(0.01)) {
                training.push_back(seq);
            } else if(rng.under(0.03)) {
                fittest.push_back(seq);
            }
        }

        return concat(create_parallel_output_tests(syms, training, Test::Training),
                      create_parallel_output_tests(syms, fittest, Test::Fittest));
*/
    }
} foo;

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
