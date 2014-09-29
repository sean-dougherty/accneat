#include "experiment.h"

#include <assert.h>
#include <regex>

using namespace NEAT;
using namespace std;

class RegexExperiment : public Experiment {
public:
    RegexExperiment()
        : Experiment("regex") {
    }

    virtual vector<Test> create_tests() override {
        const float A = 0.0;
        const float B = 1.0;

        const float S = 1.0; // Signal
        const float Q = 1.0; // Query
        const float _ = 0.0; // Null

        const real_t weight_seq = 0;
        const real_t weight_delay = 0;
        const real_t weight_query = 1;

        const char *grammar = "a+b+a+";
        regex regex_grammar{grammar};

        vector<const char *> sentences = {
            "aaa",
            "aabb",
            "bbaa",
            "aababa",
            "aababaa",
            "aaaaabbabbaaaaa",

            "aba",
            "aaba",
            "abba",
            "aabbaa",
            "aabbbba",
            "aaaaabbbbbaaaaa",
        };
        
        assert( sentences.size() % 2 == 0 );
        size_t ngrammatical = 0;
        for(const char *sentence: sentences) {
            if( regex_match(sentence, regex_grammar) )
                ngrammatical++;
        }
        assert( ngrammatical == sentences.size() / 2 );

        vector<Test> tests;

        for(const char *sentence: sentences) {
            vector<Step> steps;

            for(size_t i = 0, n = strlen(sentence); i < n; i++) {
                real_t x;

                switch(sentence[i]) {
                case 'a':
                    x = A;
                    break;
                case 'b':
                    x = B;
                    break;
                default:
                    panic();
                }

                // Create step providing signal, which has a zero output and weight
                steps.push_back({{S, x, _}, {_}, weight_seq});

                // Make a step of silence
                steps.push_back({{_, _, _}, {_}, weight_delay});
            }
            
            // End of sentence
            real_t g = regex_match(sentence, regex_grammar) ? 1.0 : 0.0;

            steps.push_back({{_, _, Q}, {g}, weight_query});

            tests.push_back({steps});
        }

        return tests;
    }
} regex_experiment;
