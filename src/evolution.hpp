#ifndef EVOLUTION_H
#define EVOLUTION_H

#include <Python.h>
#include <unordered_map>

#include "util.hpp"
#include "variation.hpp"
#include "selection.hpp"
#include "fos.hpp"
#include "complexity.hpp"
#include "globals.hpp"
#include "individual.hpp"

struct Evolution {
  
  vector<Individual*> population;
  FOSBuilder * fb = NULL;
  int gen_number = 0;
  int pop_size = 0;

  Evolution(int pop_size) {
    this->pop_size = pop_size;
    fb = new FOSBuilder();
    init_pop();
  }

  ~Evolution() {
    clear_population(population);
    if (fb)
      delete fb;
  }

  void clear_population(vector<Individual*> & population) {
    for(auto * individual : population) {
      individual->clear();
    }
    population.clear();
  }

  void init_pop() {
// ============================================================================
//     unordered_set<string> already_generated;
// ============================================================================
    population.reserve(pop_size);
    int init_attempts = 0;
    while (population.size() < pop_size) {
      auto * individual = generate_individuals(g::max_depth, g::init_strategy, g::nr_multi_trees);



// ============================================================================
//        string str_tree = tree->str_subtree();
//         if (init_attempts < g::max_init_attempts && already_generated.find(str_tree) != already_generated.end()) {
//           tree->clear();
//           init_attempts++;
//           if (init_attempts == g::max_init_attempts) {
//             print("[!] Warning: could not initialize a syntactically-unique population within ", init_attempts, " attempts");
//           }
//           continue;
//         } 
//        already_generated.insert(str_tree);
// ============================================================================

       g::fit_func->get_fitness(individual);
      population.push_back(individual);
    }
  } 

  void gomea_generation(int macro_generation) {
    // build linkage tree fos
    auto fos = fb->build_linkage_tree(population);

    // perform GOM
    vector<Individual*> offspring_population;
    offspring_population.reserve(pop_size);

    for(int i = 0; i < pop_size; i++) {
      Individual * offspring= efficient_gom(population[i], population, fos, macro_generation);
      //check_n_set_elite(offspring);
      offspring_population.push_back(offspring);
    }
    // replace parent with offspring population
    clear_population(population);

    population = offspring_population;

    ++gen_number;
  }



  void run() {

    throw runtime_error("Not implemented, please use IMS (with max runs 1 if you want a single population)");

    /*
    for(int i = 0; i < g::max_generations; i++) {
      if(g::_call_as_lib && PyErr_CheckSignals() == -1) {
        exit(1);
      }

      // update mini batch
      bool is_updated = g::fit_func->update_batch(g::batch_size);
      if (is_updated && elite) {
        elite->clear();
        elite = NULL;
        // TODO: remove elite and keep map
        for (auto it = elites_per_complexity.begin(); it != elites_per_complexity.end(); it++) {
          it->second->clear();
        }
        elites_per_complexity.clear();
      }
      
      gomea_generation();
      print("gen: ",gen_number, " elite fitness: ", elite->fitness); // TODO: remove elite
      if (converged(population, true)) {
        print("population converged");
        break;
      }

    }

    // if abs corr, append linear scaling terms
    if (g::fit_func->name() == "ac") {
      elite = append_linear_scaling(elite); 
      for (auto it = elites_per_complexity.begin(); it != elites_per_complexity.end(); it++) {
        elites_per_complexity[it->first] = append_linear_scaling(it->second);
      }
    }

    // TODO: remove this
    for (auto it = elites_per_complexity.begin(); it != elites_per_complexity.end(); it++) {
      print(it->first, " ", it->second->fitness, ":", it->second->human_repr());
    }

    if (!g::_call_as_lib) {
      print(elite->human_repr());
    }
    */
  }

};

#endif