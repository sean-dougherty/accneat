#include "organismsbuffer.h"

using namespace NEAT;
using namespace std;

OrganismsBuffer::OrganismsBuffer(rng_t &rng, size_t n)
    : _n(n) {
    _a.resize(n);
    _b.resize(n);
    _curr = &_a;

    for(auto &org: _a)
        org.genome.rng.seed(rng.integer());
    for(auto &org: _b)
        org.genome.rng.seed(rng.integer());
}

size_t OrganismsBuffer::size(){
    return _n;
}

vector<Organism> &OrganismsBuffer::curr() {
    return *_curr;
}

void OrganismsBuffer::swap() {
    if(_curr == &_a) {_curr = &_b;} else {_curr = &_a; }
    assert( _curr->size() == _n );
}
