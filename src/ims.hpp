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
  vector<int> best_sizes_discount;
  vector<float> best_kommenda_complexities;
  vector<float> best_train_mses;
  vector<float> best_val_mses;
  vector<float> times;
  string best_string;
  vector<string> best_strings;
  vector<string> MO_archive_strings;
  vector<int> consecutive_non_improvements;
  vector<float> evals;


  Evolution* evolution;
  int macro_generations = 0;

  ~IMS(){
    delete g::ea;
    delete evolution;
  }

  void run() {
    evolution = new Evolution(g::pop_size);

    g::fit_func->discount_size = g::discount_size;
    g::fit_func->change_second_obj = g::change_second_obj;
    g::mse_func->discount_size = g::discount_size;
    g::mse_func->change_second_obj = g::change_second_obj;

    auto start_time = tick();
    
    bool stop = false;
    int generations_without_improvement = 0;
    float previous_fitness = -1.;

    while(!stop) {

      // macro generation

      // update mini batch
      g::fit_func->update_batch(g::batch_size);

      bool expression_found = false;
      if(best_train_mses.size()>0){
          expression_found = best_train_mses[best_train_mses.size()-1]<1e-9;
      }
      if (expression_found || (g::max_generations > 0 && macro_generations == g::max_generations) ||
          (g::max_time > 0 && tock(start_time) >= g::max_time) || (g::max_evals > 0 && g::fit_func->evaluations>=g::max_evals) || (g::max_non_improve>0 && generations_without_improvement >= g::max_non_improve)) {
          stop = true;

          if(g::max_generations > 0 && macro_generations == g::max_generations){
              print("Stopping due to max gens");
          }
          if(g::max_time > 0 && tock(start_time) >= g::max_time){
              print("Stopping due to max time");
          }
          if(g::max_evals > 0 && g::fit_func->evaluations>=g::max_evals){
              print("Stopping due to max evals ", g::fit_func->evaluations);
          }
          if(g::max_non_improve>0 && generations_without_improvement >= g::max_non_improve){
              print("Stopping due to non improvements ", generations_without_improvement);
          }
          if(best_train_mses.size()>0 && best_train_mses[best_train_mses.size()-1]<1e-9){
              print("Stopping due to finding expression with mse 0");
          }

          break;
      }
      
      // perform generation



      if(g::use_GA){
          evolution->ga_generation(macro_generations);
      }
      else if(g::use_GP){
          evolution->gp_generation(macro_generations);
      }
      else{
          evolution->gomea_generation_MO(macro_generations);
      };

      if(g::ea->improved_this_gen){
          g::ea->improved_this_gen = false;
          generations_without_improvement = 0;
      }
      else{
          generations_without_improvement++;
      }

    if (generations_without_improvement != 0){

        if(generations_without_improvement % 5 == 0) {
            g::cmut_temp *= 0.1;
        }
    }

      // update macro gen
      macro_generations += 1;


//        if (true) {
//            print(macro_generations);
//            vector<vector<float>> fitness_per_tree;
//            fitness_per_tree.resize(g::nr_multi_trees);  // Ensure the outer vector has 4 initialized inner vectors
//
//            string str = "";
//            for (int tree_number = 0; tree_number < g::nr_multi_trees; tree_number++) {
//
//
//                for (auto ind : evolution->population) {
//
//
//                    Vec output = ind->trees[tree_number]->get_output(g::fit_func->X_train, ind->trees);
//                    float fitness = (g::fit_func->y_train - output).square().mean();
//                    fitness_per_tree[tree_number].push_back(fitness);  // Now it is safe to push_back
//                    str += to_string(fitness) + ",";
//                }
//                str.pop_back();
//                if (tree_number < g::nr_multi_trees - 1) {
//                    str += "\t";
//                } else {
//                    str += "\n";
//                }
//
//
//
//
//            }
//            ofstream csv_file;
//            csv_file.open("../nodrift.csv", ios::app);
//            csv_file << str;
//            csv_file.close();
//        }

      if(g::log){

          float best_train_mse = 9999999.;
          float best_val_mse = 9999999.;
          float best_size = 9999999.;
          float best_size_discount = 999999999.;
          float best_kommenda_complexity = 999999999.;
          string best_substring = "";

          string MO_archive_string = "{";


          bool add_comma = false;


          for (auto ind: g::ea->MO_archive) {

              float train_mse = g::fit_func->get_fitness_MO(ind, g::fit_func->X_train, g::fit_func->y_train, false)[0];

              float val_mse = g::fit_func->get_fitness_MO(ind, g::fit_func->X_val, g::fit_func->y_val, false)[0];

              int size =  ind->get_num_nodes(true, false);

              float k_complexity = ind->get_complexity_kommenda();


              if (best_train_mse > train_mse) {
;
                  best_train_mse = train_mse;
                  best_val_mse = val_mse;
                  best_size = size;
                  best_kommenda_complexity = k_complexity;
                  best_size_discount = ind->get_num_nodes(true, true);

                  best_string = "(" + to_string(ind->add) + "+(" + to_string(ind->mul) + "*"  + ind->human_repr(true) + ")" + ")";


              }
              if(add_comma){
                  MO_archive_string = MO_archive_string + ",";
              }
              else{
                  add_comma = true;
              }

              string str_ind = "(" + to_string(ind->add) + "+(" + to_string(ind->mul) + "*"  + ind->human_repr(true) + ")" + ")";
              MO_archive_string = MO_archive_string + "[" + to_string(train_mse) + "," + to_string(val_mse) + "," + to_string(ind->get_num_nodes(true)) + "," + to_string(ind->get_num_nodes(true, true)) + "," + to_string(ind->get_complexity_kommenda()) + "," + str_ind + "]";
          }
          MO_archive_string += "}";

          print(" ~ generation: ",macro_generations, " ", generations_without_improvement, " ", to_string(tock(start_time)), ", curr. best fit: ", best_train_mse, " r2 " , to_string(1. - best_train_mse/(g::fit_func->y_train - g::fit_func->y_train.mean()).square().mean()), " ", best_size, " ", best_size_discount, " ", best_kommenda_complexity, " ", best_string);

          consecutive_non_improvements.push_back(generations_without_improvement);
          best_train_mses.push_back(best_train_mse);
          best_val_mses.push_back(best_val_mse);
          best_sizes.push_back(best_size);
          best_strings.push_back(best_string);
          best_kommenda_complexities.push_back(best_kommenda_complexity);
          best_sizes_discount.push_back(best_size_discount);
          times.push_back(tock(start_time));
          MO_archive_strings.push_back(MO_archive_string);
          evals.push_back(g::fit_func->evaluations);
      }
      else{
          float best_train_mse = 9999999.;
          for (auto ind: g::ea->MO_archive) {
              float train_mse = g::fit_func->get_fitness_MO(ind, g::fit_func->X_train, g::fit_func->y_train, false)[0];

              if (best_train_mse > train_mse) {
                  best_train_mse = train_mse;
              }
          }
          best_train_mses.push_back(best_train_mse);

          print(" ~ generation: ", macro_generations, " evals ", g::fit_func->evaluations," ", best_train_mses[best_train_mses.size()-1], "", generations_without_improvement, " ", to_string(tock(start_time)), " ", g::ea->MO_archive.size(), " ", g::fit_func->evaluations);
      }
    }
    // finished

    for(int i=0;i<g::ea->MO_archive.size();i++){
        append_linear_scaling(g::ea->MO_archive[i]);
    }

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
      // 3 best size
      str += to_string(best_sizes[best_sizes.size()-1]) + "\t";
      // 4 best size
      str += to_string(best_sizes_discount[best_sizes_discount.size()-1]) + "\t";
      // 5 best size
      str += to_string(best_kommenda_complexities[best_kommenda_complexities.size()-1]) + "\t";
      // 6 best string
      str += best_string + "\t";
      // 7 train variance
      str += to_string((g::fit_func->y_train - g::fit_func->y_train.mean()).square().mean()) + "\t";
      // 8 val variance
      str += to_string((g::fit_func->y_val - g::fit_func->y_val.mean()).square().mean()) + "\t";

      csv_file << str;

      // 9 best train mse over time
      str = "";
      for(int i=0;i<best_train_mses.size()-1;i++){
          str += to_string(best_train_mses[i]) + ",";
      }
      str += to_string(best_train_mses[best_train_mses.size()-1])+"\t";
      csv_file << str;

      // 10 best val mse over time
      str = "";
      for(int i=0;i<best_val_mses.size()-1;i++){
           str += to_string(best_val_mses[i]) + ",";
      }
      str += to_string(best_val_mses[best_val_mses.size()-1])+"\t";
      csv_file << str;

      // 11 best size over time
      str = "";
      for(int i=0;i<best_sizes.size()-1;i++){
          str += to_string(best_sizes[i]) + ",";
      }
      str += to_string(best_sizes[best_sizes.size()-1])+"\t";
      csv_file << str;

        // 12 best size over time
        str = "";
        for(int i=0;i<best_sizes_discount.size()-1;i++){
            str += to_string(best_sizes_discount[i]) + ",";
        }
        str += to_string(best_sizes_discount[best_sizes_discount.size()-1])+"\t";
        csv_file << str;

        // 13 best substrings over time
        str = "";
//        for(int i=0;i<best_substrings.size()-1;i++){
//            str += best_substrings[i] + ";";
//        }
        str += "redundant\t";
        csv_file << str;

      // 14 MO over time
      str = "";
      for(int i=0;i<MO_archive_strings.size()-1;i++){
          str += MO_archive_strings[i] + ";";
      }
      str += MO_archive_strings[MO_archive_strings.size()-1]+"\t";
      csv_file << str;

      // 15 consecutive_non_improvements
      str = "";
      for(int i=0;i<consecutive_non_improvements.size()-1;i++){
          str += to_string(consecutive_non_improvements[i]) + ",";
      }
      str += to_string(consecutive_non_improvements[consecutive_non_improvements.size()-1])+"\t";
      csv_file << str;

      // 16 times
      str = "";
      for(int i=0;i<times.size()-1;i++){
          str += to_string(times[i]) + ",";
      }
      str += to_string(times[times.size()-1]);
      csv_file << str + "\t";

      // 17 evals
      str = "";
      for(int i=0;i<evals.size()-1;i++){
          str += to_string(evals[i]) + ",";
      }
      str += to_string(evals[evals.size()-1]);

      csv_file << str + "\n";

      csv_file.close();
    }
  }
};







#endif