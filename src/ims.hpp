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
  vector<int> best_sizes;
  vector<float> best_train_mses;
  vector<float> best_val_mses;
  vector<float> times;
  string best_string;
  vector<string> best_substrings;
  vector<string> MO_archive_strings;


  Evolution* evolution;
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
    evolution = new Evolution(g::pop_size);

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
          (g::max_time > 0 && tock(start_time) >= g::max_time)) {
          stop = true;

          if(g::max_generations > 0 && macro_generations == g::max_generations){
              print("Stopping due to max gens");
          }
          if(g::max_time > 0 && tock(start_time) >= g::max_time){
              print("Stopping due to max time");
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

      // update macro gen
      macro_generations += 1;

      if(g::log){
          float best_train_mse = 9999999.;
          float best_val_mse = 9999999.;
          float best_size = 9999999.;
          string best_stri = "";

          string MO_archive_string = "{";
          bool add_comma = false;
          for (auto ind: g::ea->MO_archive) {
              float val_mse = g::fit_func->get_fitness_MO(ind, g::fit_func->X_val, g::fit_func->y_val, false)[0];

              if (best_train_mse > ind->fitness[0]) {
                  best_train_mse = ind->fitness[0];
                  best_val_mse = val_mse;
                  best_size = ind->fitness[1];
                  best_stri = ind->human_repr(true);
                  best_string = best_stri;

                  vector<string> best_substrings_tmp;
                  for(int y=0; y<g::nr_multi_trees; y++){
                      best_substrings_tmp.push_back(ind->trees[y]->human_repr());
                  }
                  best_substrings = best_substrings_tmp;
              }
              if(add_comma){
                  MO_archive_string = MO_archive_string + ",";
              }
              else{
                  add_comma = true;
              }
              MO_archive_string = MO_archive_string + "[" + to_string(ind->fitness[0]) + "," + to_string(val_mse) + "," + to_string(ind->fitness[1]) + "," + ind->human_repr(true) + "]";
          }
          MO_archive_string += "}";

          print(" ~ generation: ", macro_generations, " ", to_string(tock(start_time)), ", curr. best fit: ", best_train_mse, " ", best_size, " ", best_stri);

          best_train_mses.push_back(best_train_mse);
          best_val_mses.push_back(best_val_mse);
          best_sizes.push_back(best_size);
          times.push_back(tock(start_time));
          MO_archive_strings.push_back(MO_archive_string);
      }
    }
    // finished

    if(g::log){
      ofstream csv_file;
      csv_file.open(g::csv_file, ios::app);

      string str = "";

      // 0 random state
      str += to_string(g::random_state) + "\t";
      // 1 best training mse
      str += to_string(best_train_mses[best_train_mses.size()-1]) + "\t";
      // 2 best validation mse
      str += to_string(best_val_mses[best_val_mses.size()-1]) + "\t";
      // 3 best validation mse
      str += to_string(best_sizes[best_sizes.size()-1]) + "\t";
      // 4 best string
      str += best_string + "\t";
      // 5 train variance
      str += to_string((g::fit_func->y_train - g::fit_func->y_train.mean()).square().mean()) + "\t";
      // 6 val variance
      str += to_string((g::fit_func->y_val - g::fit_func->y_val.mean()).square().mean()) + "\t";

      csv_file << str;

      // 7 best train mse over time
      str = "";
      for(int i=0;i<best_train_mses.size()-1;i++){
          str += to_string(best_train_mses[i]) + ",";
      }
      str += to_string(best_train_mses[best_train_mses.size()-1])+"\t";
      csv_file << str;

      // 8 best val mse over time
      str = "";
      for(int i=0;i<best_val_mses.size()-1;i++){
           str += to_string(best_val_mses[i]) + ",";
      }
      str += to_string(best_val_mses[best_val_mses.size()-1])+"\t";
      csv_file << str;

      // 8 best size over time
      str = "";
      for(int i=0;i<best_sizes.size()-1;i++){
          str += to_string(best_sizes[i]) + ",";
      }
      str += to_string(best_sizes[best_sizes.size()-1])+"\t";
      csv_file << str;

        // 9 best substrings over time
        str = "";
        for(int i=0;i<best_substrings.size()-1;i++){
            str += best_substrings[i] + ";";
        }
        str += best_substrings[best_substrings.size()-1]+"\t";
        csv_file << str;

      // 10 MO over time
      str = "";
      for(int i=0;i<MO_archive_strings.size()-1;i++){
          str += MO_archive_strings[i] + ";";
      }
      str += MO_archive_strings[MO_archive_strings.size()-1]+"\t";
      csv_file << str;

      // 11 times
      str = "";
      for(int i=0;i<times.size()-1;i++){
          str += to_string(times[i]) + ",";
      }
      str += to_string(times[times.size()-1]);
      csv_file << str + "\n";

      csv_file.close();
    }
  }

};







#endif