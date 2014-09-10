#pragma once

#include "neat.h"
#include <cstdlib>
#include <vector>

class rng_t {
	inline int randposneg() {
        if (rand()%2) 
            return 1; 
        else 
            return -1;
    }
    
	inline int randint(int x,int y) {
        return rand()%(y-x+1)+x;
    }

    inline double randfloat() {
        return rand() / (double) RAND_MAX;        
    }

    double gaussrand() {
        static int iset=0;
        static double gset;
        double fac,rsq,v1,v2;

        if (iset==0) {
            do {
                v1=2.0*(randfloat())-1.0;
                v2=2.0*(randfloat())-1.0;
                rsq=v1*v1+v2*v2;
            } while (rsq>=1.0 || rsq==0.0);
            fac=sqrt(-2.0*log(rsq)/rsq);
            gset=v1*fac;
            iset=1;
            return v2*fac;
        }
        else {
            iset=0;
            return gset;
        }
    }

public:
    template<typename T>
        size_t index(std::vector<T> &v, size_t begin = 0) {

        return randint(begin, v.size() - 1);
    }

    template<typename T>
        T& element(std::vector<T> &v, size_t begin = 0) {

        return v[index(v, begin)];
    }

    double prob() {
        return randfloat();
    }

    int posneg() {
        return randposneg();
    }

    double gauss() {
        return gaussrand();
    }
};
