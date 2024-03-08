#ifndef IMS_H
#define IMS_H
#include <Python.h>
#include <unordered_map>

#include "globals.hpp"
#include "util.hpp"
#include "evolution.hpp"
#include "variation.hpp"
#include "myeig.hpp"
#include "rng.hpp"
#include "individual.hpp"

#include <iostream>
#include <fstream>


using namespace std;
using namespace myeig;

struct IMS {
  //vectors for storing csv outputs
  vector<float> best_fitnesses;
  vector<int> best_sizes;
  vector<float> best_fitnesses_mse;
  vector<float> best_fitnesses_val_mse;
  vector<float> times;
  vector<float> gom_improvements;
  vector<float> coeff_improvements;
  vector<float> opt_improvements;
  vector<int> nr_improvements;
  vector<float> amount_improvements;
  vector<int> nr_unique_coeffs;
  vector<float> convergence;
  vector<int> reinject;

  Evolution* evolution = new Evolution(g::pop_size);
  int macro_generations = 0;

  ~IMS(){
    delete evolution;
  }


  bool approximately_converged(Vec & fitnesses, float upper_quantile=0.9) {
    sort(fitnesses.data(), fitnesses.data() + fitnesses.size());
    if ((fitnesses[fitnesses.size() * upper_quantile] - fitnesses[0]) < 1e-6) {
      return true;
    }
    return false;
  }

  float highest_equal_fitness(Vec & fitnesses) {
    sort(fitnesses.data(), fitnesses.data() + fitnesses.size());
    int closest_index = 0;
    for(int i=fitnesses.size()-1;i>1;i--){
      if((fitnesses[i] - fitnesses[0])<1e-6){
        closest_index = i;
        break;
      }
    }
    return static_cast<float>(closest_index)/static_cast<float>(fitnesses.size());
  }

  void run() {
    
    auto start_time = tick();
    
    bool stop = false;
    int generations_without_improvement = 0;
    float previous_fitness = -1.;

    while(!stop) {
      // macro generation

      // update mini batch
      bool mini_batch_changed = g::fit_func->update_batch(g::batch_size);
      // TODO: reevaluate archives
//      if (mini_batch_changed){
//        reevaluate_elites();
//      }

      if ((g::max_generations > 0 && macro_generations == g::max_generations) ||
          (g::max_time > 0 && tock(start_time) >= g::max_time) ||
          (g::max_evaluations > 0 && g::fit_func->evaluations >= g::max_evaluations) ||
          (g::max_node_evaluations > 0 && g::fit_func->node_evaluations >= g::max_node_evaluations)) {
          stop = true;

          if(g::max_generations > 0 && macro_generations == g::max_generations){
              print("Stopping due to max gens");
          }
          if(g::max_time > 0 && tock(start_time) >= g::max_time){
              print("Stopping due to max time");
          }
          if(g::max_evaluations > 0 && g::fit_func->evaluations >= g::max_evaluations){
              print("Stopping due to max evals");
          }

          break;
      }

      // perform generation
      if(g::MO_mode){
        evolution->gomea_generation_MO(macro_generations);
      }
      else{
        evolution->gomea_generation_SO(macro_generations);
      };



      // TODO: decide whether the evo should reinject elites

      // update macro gen
      macro_generations += 1;

      float best = 9999999.;
      float best_2 = 9999999.;
      string best_stri = "";


      if(g::MO_mode) {
          for (auto ind: g::ea->MO_archive) {
              if (best > ind->fitness[0]) {
                  best = ind->fitness[0];
                  best_2 = ind->fitness[1];
                  best_stri = ind->human_repr(true);
              }
          }
      }
      else{
          for (auto ind: g::ea->SO_archive) {
              if (best > ind->fitness[0]) {
                  best = ind->fitness[0];
                  best_2 = ind->fitness[1];
                  best_stri = ind->human_repr(true);
              }
          }
      }

      float best_pop = 9999999.;
      float best_2_pop = 9999999.;
      string best_stri_pop = "";
      int nis = 9999;

      for(auto ind: evolution->population){
          if(best_pop>ind->fitness[0]){
              best_pop = ind->fitness[0];
              best_2_pop = ind->fitness[1];
              best_stri_pop = ind->human_repr(false);
              nis = ind->NIS;
          }
      }

      print(" ~ generation: ", macro_generations, " ", to_string(tock(start_time)), ", curr. best fit: ", best, " ", best_2, " ", best_stri, " ", best_pop, " ", best_2_pop, " ", best_stri_pop, " ", nis);

    }
    // finished

    if(g::log){
      ofstream csv_file;
      csv_file.open(g::csv_file, ios::app);

      string str = "";

      csv_file << str;
      csv_file.close();
    }
  }

};







#endif