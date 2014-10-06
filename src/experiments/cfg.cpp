#include "std.h" // Must be included first. Precompiled header with standard library includes.
#include "experiment.h"
#include <assert.h>

using namespace NEAT;
using namespace std;

static vector<Test> create_tests(const vector<string> &sentences,
                                 const vector<bool> &is_grammatical);

static class Cfg_XSX : public Experiment {
public:
    Cfg_XSX() : Experiment("cfg-XSX") {
    }

    virtual vector<Test> create_tests() override {
        // S -> aSa
        // S -> bSb
        // S -> \0
        vector<string> sentences;
        append(sentences, permute_repeat("ab", 2));
        append(sentences, permute_repeat("ab", 4));

        vector<bool> is_grammatical;
        for(string &s: sentences) {
            size_t n = s.size();
            if(n % 2 != 0) {
                is_grammatical.push_back(false);
            } else {
                bool mirrored = true;
                for(size_t i = 0; mirrored && (i < n/2); i++) {
                    mirrored = s[i] == s[n - 1 - i];
                }
                is_grammatical.push_back(mirrored);
            }
        }
/*
        size_t ngrammatical = 0;
        size_t nungrammatical = 0;
        for(bool x: is_grammatical) {
            if(x) {
                ngrammatical++;
            } else {
                nungrammatical++;
            }
        }

        rng_t rng;
        while( sentences.size() > 32 ) {
            int i = rng.integer(0, sentences.size() - 1);
            if(!is_grammatical[i]) {
                if( nungrammatical >= ngrammatical ) {
                    sentences.erase( sentences.begin() + i );
                    is_grammatical.erase( is_grammatical.begin() + i );
                    nungrammatical--;
                }
            } else {
                if( ngrammatical >= nungrammatical ) {
                    sentences.erase( sentences.begin() + i );
                    is_grammatical.erase( is_grammatical.begin() + i );
                    ngrammatical--;
                }
            }
        }
*/
        for(size_t i = 0; i < sentences.size(); i++) {
            cout << sentences[i] << ": " << is_grammatical[i] << endl;
        }

        return ::create_tests(sentences, is_grammatical);
    }
} cfg_XSX;

static vector<Test> create_tests(const vector<string> &sentences,
                                 const vector<bool> &is_grammatical) {
    const real_t A[] = {0.0, 0.0};
    const real_t B[] = {0.0, 1.0};
    const real_t C[] = {1.0, 0.0};
    const real_t D[] = {1.0, 1.0};

    const real_t S = 1.0; // Signal
    const real_t Q = 1.0; // Query
    const real_t _ = 0.0; // Null

    const real_t weight_seq = 0;
    const real_t weight_delay = 0;

    size_t ncorrect = 0;
    for(bool x: is_grammatical) {
        if(x) {
            ncorrect++;
        }
    }
    cout << "ncorrect = " << ncorrect << " / " << sentences.size() << endl;

    const real_t weight_query_correct = 1.0 / ncorrect;
    const real_t weight_query_incorrect = 1.0 / (sentences.size() - ncorrect);
        
    vector<Test> tests;
    for(size_t isentence = 0; isentence < sentences.size(); isentence++) {
        const string &sentence = sentences[isentence];
        vector<Step> steps;

        for(size_t ielement = 0; ielement < sentence.size(); ielement++) {
            const real_t *X;

            switch(sentence[ielement]) {
            case 'a':
                X = A;
                break;
            case 'b':
                X = B;
                break;
            case 'c':
                X = C;
                break;
            case 'd':
                X = D;
                break;
            default:
                panic();
            }

            real_t x = X[0];
            real_t y = X[1];

            // Create step providing signal, which has a zero output and weight
            steps.push_back({{S, x, y, _}, {_}, weight_seq});

            // Make a step of silence
            steps.push_back({{_, _, _, _}, {_}, weight_delay});
        }
            
        // End of sentence
        if( is_grammatical[isentence] ) {
            steps.push_back({{_, _, _, Q}, {1.0}, weight_query_correct});
        } else {
            steps.push_back({{_, _, _, Q}, {0.0}, weight_query_incorrect});
        }

        tests.push_back({steps});
    }

    return tests;
}
