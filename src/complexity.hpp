#ifndef COMPLEXITY_H
#define COMPLEXITY_H

#include "individual.hpp"
#include "operator.hpp"
#include "globals.hpp"

float compute_complexity(Individual * individual) {
  if (g::complexity_type == "node_count") {
    return individual->get_num_nodes(true);
  } 
  throw std::runtime_error("Unrecognized complexity type: " + g::complexity_type);
}


#endif