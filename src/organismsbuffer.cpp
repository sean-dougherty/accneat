#include "organismsbuffer.h"

using namespace NEAT;
using namespace std;

OrganismsBuffer::OrganismsBuffer(rng_t &rng, size_t n)
    : _n(n) {
    _a.resize(n);
    _b.resize(n);
    _curr = &_a;

    for(size_t i = 0; i < n; i++) {
        _a[i].population_index = i;
        _a[i].genome.rng.seed(rng.integer());
    }
    for(size_t i = 0; i < n; i++) {
        _b[i].population_index = i;
        _b[i].genome.rng.seed(rng.integer());
    }
}

size_t OrganismsBuffer::size(){
    return _n;
}

vector<Organism> &OrganismsBuffer::curr() {
    return *_curr;
}

void OrganismsBuffer::next_generation(int generation) {
    if(_curr == &_a) {_curr = &_b;} else {_curr = &_a; }
    assert( _curr->size() == _n );

    for(Organism &org: curr())
        org.init(generation);

}
