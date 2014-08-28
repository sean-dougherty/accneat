#pragma once

#include "population.h"
#include "organism.h"

NEAT::Population *polylogic_test(int gens);
bool polylogic_evaluate(NEAT::Organism *org);
int polylogic_epoch(NEAT::Population *pop,int generation,char *filename, int &winnernum, int &winnergenes,int &winnernodes);
