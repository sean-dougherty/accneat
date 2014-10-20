#include "std.h" // Must be included first. Precompiled header with standard library includes.
#include "batteryexperiment.h"
#include "util.h"
#include <assert.h>

using namespace NEAT;
using namespace std;

static vector<Test> create_tests_1bit(const char *grammar,
                                      const vector<string> &sentences);
static vector<Test> create_tests_2bit(const char *grammar,
                                      const vector<string> &sentences);

static class Regex_aba : public BatteryExperiment {
public:
    Regex_aba() : BatteryExperiment("regex-aba") {
    }

    virtual vector<Test> create_tests() override {
        const char *grammar = "a+b+a+";

        vector<string> sentences = {
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

        return ::create_tests_1bit(grammar, sentences);
    }
} regex_aba;

static class Regex_aba_2bit : public BatteryExperiment {
public:
    Regex_aba_2bit() : BatteryExperiment("regex-aba-2bit") {
    }

    virtual vector<Test> create_tests() override {
        const char *grammar = "a+b+a+";

        vector<string> sentences = {
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

        return ::create_tests_2bit(grammar, sentences);
    }
} regex_aba_2bit;

static class Regex_XYXY : public BatteryExperiment {
public:
    Regex_XYXY() : BatteryExperiment("regex-XYXY") {
    }

    virtual vector<Test> create_tests() override {
        const char *grammar = "[ad][bc][ad][bc]";

        vector<string> sentences = permute_repeat("abcd", 4);

        return ::create_tests_2bit(grammar, sentences);
    }
} regex_XYXY;

static vector<Test> create_tests_1bit(const char *grammar,
                                      const vector<string> &sentences) {
    const real_t A = 0.0;
    const real_t B = 1.0;

    const real_t S = 1.0; // Signal
    const real_t Q = 1.0; // Query
    const real_t _ = 0.0; // Null

    const real_t weight_seq = 0;
    const real_t weight_delay = 0;
    const real_t weight_query = 1;

    regex regex_grammar{grammar};
        
    vector<Test> tests;
    for(const string &sentence: sentences) {
        vector<Step> steps;

        for(size_t i = 0, n = sentence.size(); i < n; i++) {
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

static vector<Test> create_tests_2bit(const char *grammar,
                                      const vector<string> &sentences) {
    const real_t A[] = {0.0, 0.0};
    const real_t B[] = {0.0, 1.0};
    const real_t C[] = {1.0, 0.0};
    const real_t D[] = {1.0, 1.0};

    const real_t S = 1.0; // Signal
    const real_t Q = 1.0; // Query
    const real_t _ = 0.0; // Null

    const real_t weight_seq = 0;
    const real_t weight_delay = 0;

    regex regex_grammar{grammar};

    size_t ncorrect = 0;
    for(const string &sentence: sentences) {
        if(regex_match(sentence, regex_grammar)) {
            ncorrect++;
        }
    }
    cout << "ncorrect = " << ncorrect << " / " << sentences.size() << endl;

    const real_t weight_query_correct = 1.0 / ncorrect;
    const real_t weight_query_incorrect = 1.0 / (sentences.size() - ncorrect);
        
    vector<Test> tests;
    for(const string &sentence: sentences) {
        vector<Step> steps;

        for(size_t i = 0, n = sentence.size(); i < n; i++) {
            const real_t *X;

            switch(sentence[i]) {
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
        if( regex_match(sentence, regex_grammar) ) {
            steps.push_back({{_, _, _, Q}, {1.0}, weight_query_correct});
        } else {
            steps.push_back({{_, _, _, Q}, {0.0}, weight_query_incorrect});
        }

        tests.push_back({steps});
    }

    return tests;
}
