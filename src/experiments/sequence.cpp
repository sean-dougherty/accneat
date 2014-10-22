#include "std.h" // Must be included first. Precompiled header with standard library includes.
#include "batteryexperiment.h"
#include "util.h"
#include <assert.h>

using namespace NEAT;
using namespace std;

static vector<Test> create_parallel_output_tests(string syms,
                                                 vector<string> &sequences);

struct Seq1bit2elExperiment : public BatteryExperiment {
    Seq1bit2elExperiment() : BatteryExperiment("seq-1bit-2el") {
    }

    virtual vector<Test> create_tests() override {
        string syms = "ab";
        vector<string> seqs = permute_repeat(syms, 2);

        return create_parallel_output_tests(syms, seqs);
    }
} seq_1bit_2el;

struct Seq1bit3elExperiment : public BatteryExperiment {
    Seq1bit3elExperiment() : BatteryExperiment("seq-1bit-3el") {
    }

    virtual vector<Test> create_tests() override {
        string syms = "ab";
        vector<string> seqs = permute_repeat(syms, 3);

        return create_parallel_output_tests(syms, seqs);
    }
} seq_1bit_3el;

struct Seq1bit4elExperiment : public BatteryExperiment {
    Seq1bit4elExperiment() : BatteryExperiment("seq-1bit-4el") {
    }

    virtual vector<Test> create_tests() override {
        string syms = "ab";
        vector<string> seqs = permute_repeat(syms, 4);

        return create_parallel_output_tests(syms, seqs);
    }
} seq_1bit_4el;

static vector<Test> create_parallel_output_tests(string syms,
                                                 vector<string> &sequences) {
    const real_t weight_seq = 5;
    const real_t weight_query = 50;

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

                steps.emplace_back(input, output, weight_seq);
            }
            
            //Create silence
            {
                vector<real_t> input;
                append(input, _); // No symbol this step
                append(input, _); // Not querying
                append(input, _, nbits); // Empty symbol

                vector<real_t> output;
                append(output, _, sequence_len * nbits); // Empty output

                steps.emplace_back(input, output, weight_seq);
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

            steps.emplace_back(input, output, weight_query);
        }

        tests.emplace_back(sequence, steps);
    }

    return tests;
}
