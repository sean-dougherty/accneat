#include "rng.h"

#include <assert.h>
#include <cstdlib>
#include <iostream>

using namespace NEAT;
using namespace std;

static bool equals(double x, double y) {
    return abs((x) - (y)) / double(y) < 0.01;
}

#define assert_equals(x, y)                                             \
    if( !equals(x,y) ) {                                                \
        cerr << __FILE__ << ":" << __LINE__ << ": " << x << " != " << y << endl; \
        exit(1);                                                        \
    }

#define assert_equals_vec(x, y)                 \
    for(size_t i = 0; i < x.size(); i++) {      \
        assert_equals(x[i], y[i]);              \
    }

#define assert_nequals_vec(x, y)                                        \
    for(size_t i = 0; i < x.size(); i++) {                              \
        if(!equals(x[i], y[i])) break;                                  \
        if(i == x.size() - 1) {                                         \
            cerr << __FILE__ << ":" << __LINE__ << ": vectors equal" << endl; \
            exit(1);                                                    \
        }                                                               \
    }

void rng_t::test() {
    // seed
    {
        rng_t rng1;
        rng_t rng2;

        {
            vector<double> x, y;

            rng1.seed(1);
            x = {rng1.prob(), rng1.prob(), rng1.prob()};
            rng2.seed(1);
            y = {rng2.prob(), rng2.prob(), rng2.prob()};
            assert_equals_vec(x, y);
        }

        {
            vector<double> x, y;

            rng1.seed(2);
            x = {rng1.prob(), rng1.prob(), rng1.prob()};
            rng2.seed(3);
            y = {rng2.prob(), rng2.prob(), rng2.prob()};
            assert_nequals_vec(x, y);
        }

    }

    // element
    {
        const size_t N = 1000000;

        rng_t rng;
        vector<size_t> vec = {0, 0, 0, 0, 0};
        for(size_t i = 0; i < N; i++) {
            rng.element(vec)++;
        }

        for(size_t x: vec) {
            assert_equals( double(x) / N, 1.0 / vec.size() );
        }
    }

    // prob
    {
        const size_t N = 1000000;
        const size_t NBINS = 5;

        rng_t rng;
        size_t count[NBINS] = {0};
        
        for(size_t i = 0; i < N; i++) {
            double x = rng.prob();
            
            assert(x >= 0.0 && x <= 1.0);

            size_t bin = min(x * NBINS, double(NBINS - 1));

            count[bin]++;
        }

        for(auto n: count) {
            assert_equals( n/double(N), 1.0/double(NBINS));
        }
    }

    // posneg
    {
        const size_t N = 1000000;

        rng_t rng;
        size_t count[] = {0, 0};

        for(size_t i = 0; i < N; i++) {
            int x = rng.posneg();

            assert(x == 1 || x == -1);

            count[(x + 1) / 2]++;
        }

        assert_equals( double(count[0]) / N, 0.5);
    }

    // gauss
    {
        const size_t N = 10000000;
        const size_t NBINS = 3;
        
        rng_t rng;
        size_t count_neg[NBINS] = {0};
        size_t count_pos[NBINS] = {0};

        for(size_t i = 0; i < N; i++) {
            double x = rng.gauss();
            size_t *count = x < 0 ? count_neg : count_pos;

            size_t bin = min(size_t(abs(x)), NBINS - 1);
            count[bin]++;
        }

        assert_equals( double(count_neg[0]) / N, 0.3413 );
        assert_equals( double(count_neg[1]) / N, 0.4772 - 0.3413 );
        assert_equals( double(count_neg[2]) / N, 0.5 - 0.4772 );
        assert_equals( double(count_pos[0]) / N, 0.3413 );
        assert_equals( double(count_pos[1]) / N, 0.4772 - 0.3413 );
        assert_equals( double(count_pos[2]) / N, 0.5 - 0.4772 );
    }

    cout << "rng test passed" << endl;
}
