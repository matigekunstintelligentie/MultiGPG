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
  
  float mse_before_optimisation = -1;
  float r2_before_optimisation = -1;
  float mse_before_ls = -1;
  float r2_before_ls = -1;


  int MAX_POP_SIZE = (int) pow(2,20);
  int SUB_GENs = 4;

  vector<Evolution*> evolutions;
  int macro_generations = 0;
  unordered_map<float, Individual*> elites_per_complexity;

  ~IMS() {
    for (Evolution * e : evolutions) {
      delete e;
    }
    reset_elites();
  }

  Individual * select_elite(float rel_compl_importance=0.0) {
    // get relative fitness among elites
    float min_fit = INF;
    float max_fit = NINF;
    float min_compl = INF;
    float max_compl = NINF;
    vector<Individual*> ordered_elites; ordered_elites.reserve(elites_per_complexity.size());
    vector<float> ordered_fitnesses; ordered_fitnesses.reserve(elites_per_complexity.size());
    vector<float> ordered_complexities; ordered_complexities.reserve(elites_per_complexity.size());

    for(auto it = elites_per_complexity.begin(); it != elites_per_complexity.end(); ++it) {

      float f = it->second->fitness[0];
      float c = it->first;
      
      if (f > max_fit)
        max_fit = f;
      if (f < min_fit)
        min_fit = f;
      if (c > max_compl)
        max_compl = c;
      if (c < min_compl)
        min_compl = c;
      
      ordered_elites.push_back(it->second);
      ordered_fitnesses.push_back(f);
      ordered_complexities.push_back(c);
    }

    // get best
    int best_idx = 0;
    float best_score = 0;
    for(int i = 0; i < ordered_fitnesses.size(); i++) {
      // normalize fitness & compl (with higher=better)
      float fit_score = 1.0 - (ordered_fitnesses[i] - min_fit)/(max_fit - min_fit);
      float compl_score = 1.0 - (ordered_complexities[i] - min_compl)/(max_compl - min_compl);

      // incl. penalty
      float score = (1.0 - rel_compl_importance) * fit_score + rel_compl_importance * compl_score;
      if (score > best_score) {
        best_score = score;
        best_idx = i;
      }
    }
    return ordered_elites[best_idx];
  }

  bool initialize_new_evolution() {
    // if it is the first evolution
    int pop_size;
    if (evolutions.empty()) {
      evolutions.reserve(10);
      pop_size = g::pop_size;
    } else {
      pop_size = evolutions[evolutions.size()-1]->population.size() * 2;
    }
    // skip if new pop.size is too large
    if (pop_size > MAX_POP_SIZE) {
      return false;
    }
    // or skip if options set not to use IMS and we already have 1 evolution
    if (g::disable_ims && evolutions.size() > 0) {
      return false;
    }
    Evolution * evo = new Evolution(pop_size);
    if (g::disable_ims && elites_per_complexity.size() > 0) {
      // if this was a re-start of the single population that converged before
      // inject a random elite by replacing a random solution
      if(g::reinject_elite){
          // get random elite
          auto it = elites_per_complexity.begin();
          std::advance(it, Rng::randi(elites_per_complexity.size()));
          Individual * an_elite = it->second;
          
          int repl_idx = Rng::randi(evo->population.size());
          evo->population[repl_idx]->clear();
          evo->population[repl_idx] = an_elite->clone();
          //print(" + injecting an elite into re-started population");
          reinject.push_back(macro_generations);
      }
    }
    evolutions.push_back(evo);
    //print(" + init. new evolution with pop.size: ",pop_size);
    return true;
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

  void terminate_obsolete_evolutions() {
    int largest_obsolete_idx = -1;
    for(int i = evolutions.size() - 1; i >= 0; i--) {
      auto fitnesses_i = g::fit_func->get_fitnesses(evolutions[i]->population, false);
      convergence.push_back(highest_equal_fitness(fitnesses_i));

      float med_fit_i = median(fitnesses_i);

      // if there is only one evolution & it converged, terminate it
      if (g::disable_ims && approximately_converged(fitnesses_i)) {
        largest_obsolete_idx = i;
      }

      for (int j = i-1; j >= 0; j--) {
        auto fitnesses_j = g::fit_func->get_fitnesses(evolutions[j]->population, false);
        float med_fit_j = median(fitnesses_j);
        if (med_fit_j > med_fit_i || approximately_converged(fitnesses_j)) {
          // will have to terminate j and previous
          largest_obsolete_idx = j;
          break;
        }
      }
      // got something, stop checking
      if (largest_obsolete_idx >= 0) {
        //print(" - terminating evolutions with pop.size <= ", evolutions[largest_obsolete_idx]->pop_size);
        break;
      }
    }

    // terminate all previous
    for (int i = 0; i <= largest_obsolete_idx; i++) {
      // free memory
      delete evolutions[i];
    }
    // resize evolutions array
    if (largest_obsolete_idx > -1) {
      evolutions = vector<Evolution*>(evolutions.begin() + largest_obsolete_idx + 1, evolutions.end());
    }
  }

  void reset_elites() {
    for(auto it = elites_per_complexity.begin(); it != elites_per_complexity.end(); it++) {
      it->second->clear();
    }
    elites_per_complexity.clear();
  }

  void reevaluate_elites() {
    for(auto it = elites_per_complexity.begin(); it != elites_per_complexity.end(); it++) {
      g::fit_func->get_fitness(it->second);
    }
  }

  void update_elites(vector<Individual*>& population) {
    for (Individual * tree : population){
      float c = compute_complexity(tree);
      // determine if to insert this among elites and eliminate now-obsolete elites
      bool worse_or_equal_than_existing = false;
      vector<float> obsolete_complexities; obsolete_complexities.reserve(elites_per_complexity.size());
      for(auto it = elites_per_complexity.begin(); it != elites_per_complexity.end(); it++) {
        if (c >= it->first && tree->fitness >= it->second->fitness) {
          // this tree is equal or worse than an existing elite
          worse_or_equal_than_existing = true;
          break;
        }
        // check if a previous elite became obsolete
        if (c <= it->first && tree->fitness < it->second->fitness) {
          obsolete_complexities.push_back(it->first);
        }  
      }
      if (worse_or_equal_than_existing)
        continue;
      // remove obsolete elites
      for(float oc : obsolete_complexities) {
        elites_per_complexity[oc]->clear();
        elites_per_complexity.erase(oc);
      }
      // save this tree as a new elite
      elites_per_complexity[c] = tree->clone();
      //print("\tfound new equation with fitness ", tree->fitness, " and complexity ", c);
    }
  }



  void run() {
    
    auto start_time = tick();

    // initialize the first evolution
    initialize_new_evolution();
    
    bool stop = false;
    int generations_without_improvement = 0;
    float previous_fitness = -1.;
    while(!stop) {
      // macro generation

      // update mini batch
      bool mini_batch_changed = g::fit_func->update_batch(g::batch_size);
      if (mini_batch_changed){
        reevaluate_elites();
      }

      int curr_num_evos = evolutions.size();
      for (int i = 0; i < curr_num_evos + 1; i++) {
        // check should stop
        if(g::_call_as_lib && PyErr_CheckSignals() == -1) {
          exit(1);
        }
// ============================================================================
//         auto fitnesses_i = g::fit_func->get_fitnesses(evolutions[0]->population, false);
//         || approximately_converged(fitnesses_i)
// ============================================================================
        if (
          (g::max_generations > 0 && macro_generations == g::max_generations) ||
          (g::max_time > 0 && tock(start_time) >= g::max_time) ||
          (g::max_evaluations > 0 && g::fit_func->evaluations >= g::max_evaluations) ||
          (g::max_node_evaluations > 0 && g::fit_func->node_evaluations >= g::max_node_evaluations) 
        ) {
          stop = true;
          break;
        }

        // find evo that must perform a generation
        bool should_perform_gen = false;
        if (i == 0 || evolutions[i-1]->gen_number > 0 && evolutions[i-1]->gen_number % SUB_GENs == 0){
          should_perform_gen = true;
          if (i > 0){
            evolutions[i-1]->gen_number = 0; // reset counter
          }
        }

        if (!should_perform_gen)
          continue;

         // must be initialized
         if (i == evolutions.size()) {
           bool possible = initialize_new_evolution();
           if (!possible)
             continue;
         }

        //  if(macro_generations==0){
        //   auto fitnesses_i = g::fit_func->get_fitnesses(evolutions[evolutions.size()-1]->population, false);
        //   ofstream csv_file;
        //   csv_file.open("./results/firstfitness.csv", ios::app);
          
        //   string str = "";
        //   for(int i=0;i<fitnesses_i.size()-1;i++){
        //     str += to_string(fitnesses_i[i]) + ","; 
        //   }
        //   str += to_string(fitnesses_i[fitnesses_i.size()-1])+"\n";
        //   csv_file << str;
        //   csv_file.close();
        // }


        // perform generation
        evolutions[i]->gomea_generation(macro_generations);


        // update elites
        update_elites(evolutions[i]->population);

        //print("\tperformed evo with pop.size: ",evolutions[i]->pop_size);
      }

      // decide if some evos should terminate
      terminate_obsolete_evolutions();

      // update macro gen
      macro_generations += 1;
      
// ============================================================================
// ============================================================================
//       if(macro_generations==1 || macro_generations==10){
//            auto fitnesses_i = g::fit_func->get_fitnesses(evolutions[evolutions.size()-1]->population, false);
//            ofstream csv_file;
//            csv_file.open("./results/firstfitness.csv", ios::app);
//         
//            string str = "";
//            for(int i=0;i<fitnesses_i.size()-1;i++){
//              str += to_string(fitnesses_i[i]) + ","; 
//            }
//            str += to_string(fitnesses_i[fitnesses_i.size()-1])+"\n";
//            csv_file << str;
//            csv_file.close();
//            if(macro_generations==10){
//              throw runtime_error("Not how you should program");
//            }
// 
//       }
// ============================================================================
// ============================================================================
      
      
      Individual * curr_elite = select_elite(0.0)->clone();
      float curr_best_fit = g::fit_func->get_fitness(curr_elite, g::fit_func->X_train, g::fit_func->y_train);
      
      
// ============================================================================
//       if(g::optimise_after){
//           bool prev_state = g::use_optimiser;
//           g::use_optimiser = true;
//           if(g::optimiser_choice=="lm"){
//             g::lm_max_fev = 100;
//             coeff_opt_lm(curr_elite, false);
//             g::lm_max_fev = 10;
//           }
//           if(g::optimiser_choice=="bfgs"){
//             g::bfgs_max_iter = 10;
//             coeff_opt_bfgs(curr_elite, false);
//             g::bfgs_max_iter = 5;
//           }
//           g::use_optimiser = prev_state;
//           g::fit_func->get_fitness(curr_elite);
//           curr_best_fit = curr_elite->fitness;
//       }
// 
// ============================================================================
      
      if(previous_fitness<0){
          previous_fitness = curr_best_fit;
      }
      else{
          if(previous_fitness<=curr_best_fit){
              generations_without_improvement += 1;
          }
          else{
              generations_without_improvement = 0;
          }
          previous_fitness = curr_best_fit;
      }
      
      if (generations_without_improvement != 0){
        
        if(generations_without_improvement % 5 == 0) {
          g::cmut_temp *= 0.1;
        }
      }

      print(" ~ macro generation: ", macro_generations, ", curr. best fit: ",curr_best_fit);
      if(g::log){
// ============================================================================
//         if(g::fit_func->name()=="mse"){
//           Node * lsind = append_linear_scaling(curr_elite);
//           curr_best_fit = g::fit_func->get_fitness(lsind);
//         }
// ============================================================================


        //print(curr_elite->human_repr());


        best_fitnesses.push_back(1. - curr_best_fit/(g::fit_func->y_train - g::fit_func->y_train.mean()).square().mean());
        best_sizes.push_back(curr_elite->get_num_nodes(true));
        best_fitnesses_mse.push_back(curr_best_fit);
        best_fitnesses_val_mse.push_back(g::fit_func->get_fitness(select_elite(0.0), g::mse_func->X_val, g::mse_func->y_val));

        //print("nr improvements: ", g::nr_improvements, " ", g::amount_improvements);

        nr_improvements.push_back(g::nr_improvements);
        amount_improvements.push_back(g::amount_improvements);

        g::nr_improvements = 0;
        g::amount_improvements = 0.;

        times.push_back(tock(start_time));

        }

      curr_elite->clear();
    }


    // finished

    // if abs corr, append linear scaling terms
    if (g::fit_func->name() == "ac" || g::fit_func->name()=="lsmse"){
      for (auto it = elites_per_complexity.begin(); it != elites_per_complexity.end(); it++) {
        Node * lin_scale = append_linear_scaling(it->second->trees[it->second->trees.size()-1], it->second->trees);
        it->second->trees[it->second->trees.size()-1] = lin_scale;
        elites_per_complexity[it->first] = it->second;
      }
    }

    if (!g::_call_as_lib) { // TODO: remove false
      print("\nAll elites found:");
      for (auto it = elites_per_complexity.begin(); it != elites_per_complexity.end(); it++) {
        print(it->first, " ", it->second->fitness[0], ":", it->second->human_repr(true));
      }
      print("\nBest w.r.t. complexity for chosen importance:");
      print(this->select_elite(g::rel_compl_importance)->human_repr(true));
        print(this->select_elite(g::rel_compl_importance)->human_repr(false));
    }

    if(g::log){
      ofstream csv_file;
      csv_file.open(g::csv_file, ios::app);

      string str = "";
      for(int i=0;i<best_fitnesses.size()-1;i++){
        str += to_string(best_fitnesses[i]) + ","; 
      }
      str += to_string(best_fitnesses[best_fitnesses.size()-1])+"\t";

      for(int i=0;i<best_sizes.size()-1;i++){
        str += to_string(best_sizes[i]) + ","; 
      }
      str += to_string(best_sizes[best_sizes.size()-1])+"\n";

      csv_file << str;
      csv_file.close();
    }

// ============================================================================
//     if(g::log){
// 
//       print("JAC evals "+ to_string(g::jacobian_evals));
//       ofstream csv_file;
//       csv_file.precision(NUM_PRECISION);
//       csv_file.open(g::csv_file, ios::app);
// 
//       Individual * elite_clone = select_elite(0.0)->clone();
// 
//       print("fitness", to_string(g::mse_func->get_fitness(elite_clone, g::mse_func->X_train, g::mse_func->y_train)));
//       
//       string str = "";
// 
//       // 0 random state
//       str += to_string(g::random_state) + "\t";
//       // 1 expression
//       str += elite_clone->human_repr()+"\t";
//       // 2 mse fitness
// 
//       str += to_string(g::mse_func->get_fitness(elite_clone, g::mse_func->X_train, g::mse_func->y_train))+"\t";
// 
//       // 3 r2 fitness
//       str += to_string((g::fit_func->y_train - g::fit_func->y_train.mean()).square().mean())+"\t";
//       // 4 val mse fitness
//       str += to_string(g::mse_func->get_fitness(elite_clone, g::mse_func->X_val, g::mse_func->y_val))+"\t";
//       // 5 complexity
//       str += to_string(compute_complexity(elite_clone))+"\t";
//       // 6 fitness evals
//       str += to_string(g::fit_func->evaluations)+"\t";
//       // 7 mse evals
//       str += to_string(g::mse_func->opt_evaluations)+"\t";
//       // 8 jac evals
//       str += to_string(g::jacobian_evals)+"\t";
// 
// 
// 
//       Individual * bfgs_clone = elite_clone->clone();
// 
//       g::use_optimiser = true;
//       g::bfgs_max_iter = 100;
//       coeff_opt_bfgs(bfgs_clone, false);
// 
//       // 9 bfgs expression
//       str += bfgs_clone->human_repr()+"\t";
//       // 10 bfgs train mse
//       str += to_string(g::mse_func->get_fitness(bfgs_clone, g::mse_func->X_train, g::mse_func->y_train))+"\t";
// 
//       // 11 bfgs val mse
//       str += to_string(g::mse_func->get_fitness(bfgs_clone, g::mse_func->X_val, g::mse_func->y_val))+"\t";
// 
//       bfgs_clone->clear();
// 
//       Individual * lm_clone = elite_clone->clone();
// 
//       g::use_optimiser = true;
//       g::lm_max_fev = 100;
//       coeff_opt_bfgs(lm_clone, false);
// 
//       // 12 lm expression
//       str += lm_clone->human_repr()+"\t";
//       // 13 lm train mse
//       str += to_string(g::mse_func->get_fitness(lm_clone, g::mse_func->X_train, g::mse_func->y_train))+"\t";
//       // 14 lm val mse
//       str += to_string(g::mse_func->get_fitness(lm_clone, g::mse_func->X_val, g::mse_func->y_val))+"\t";
// 
//       lm_clone->clear();
// 
//       Individual * ls_clone = elite_clone->clone();
//       ls_clone = append_linear_scaling(ls_clone);
// 
//       // 15 ls mse
//       str += ls_clone->human_repr()+"\t";
//       // 16 ls train mse
//       str += to_string(g::mse_func->get_fitness(ls_clone, g::mse_func->X_train, g::mse_func->y_train))+"\t";
//       // 17 ls val mse
//       str += to_string(g::mse_func->get_fitness(ls_clone, g::mse_func->X_val, g::mse_func->y_val))+"\t";
// 
//       ls_clone->clear();
//       
//       // 18 mse fitnesses
//       for(int i=0;i<best_fitnesses.size()-1;i++){
//         str += to_string(best_fitnesses[i]) + ","; 
//       }
//       str += to_string(best_fitnesses[best_fitnesses.size()-1])+"\t";
// 
//       // 19 mse val fitnesses
//       for(int i=0;i<best_fitnesses_mse.size()-1;i++){
//         str += to_string(best_fitnesses_mse[i]) + ","; 
//       }
//       str += to_string(best_fitnesses_mse[best_fitnesses_mse.size()-1])+"\t";
// 
//       for(int i=0;i<best_fitnesses_val_mse.size()-1;i++){
//         str += to_string(best_fitnesses_val_mse[i]) + ","; 
//       }
//       str += to_string(best_fitnesses_val_mse[best_fitnesses_val_mse.size()-1])+"\t"; 
//       csv_file << str;
// 
//       str = "";
//       for(int i=0;i<gom_improvements.size()-1;i++){
//         str += to_string(gom_improvements[i]) + ","; 
//       }
//       str += to_string(gom_improvements[gom_improvements.size()-1])+"\t";
//       csv_file << str;
// 
//       str = "";
//       for(int i=0;i<coeff_improvements.size()-1;i++){
//         str += to_string(coeff_improvements[i]) + ","; 
//       }
//       str += to_string(coeff_improvements[coeff_improvements.size()-1])+"\t";
//       csv_file << str;
// 
//       str = "";
//       for(int i=0;i<opt_improvements.size()-1;i++){
//         str += to_string(opt_improvements[i]) + ","; 
//       }
//       str += to_string(opt_improvements[opt_improvements.size()-1])+"\t";
//       csv_file << str;
//       
//       str = "";
//       for(int i=0;i<nr_improvements.size()-1;i++){
//         str += to_string(nr_improvements[i]) + ","; 
//       }
//       str += to_string(nr_improvements[nr_improvements.size()-1])+"\t";
//       csv_file << str;
// 
//       str = "";
//       for(int i=0;i<amount_improvements.size()-1;i++){
//         str += to_string(amount_improvements[i]) + ","; 
//       }
//       str += to_string(amount_improvements[amount_improvements.size()-1])+"\t";
//       csv_file << str;
//       
//       str = "";
//       if(reinject.size()>0){
//         for(int i=0;i<reinject.size()-1;i++){
//           str += to_string(reinject[i]) + ","; 
//         }
//       str += to_string(reinject[reinject.size()-1])+"\t";
//       }
//       else{
//         str += "\t";
//       }
//       
//       csv_file << str;
// 
//       str = "";
//       for(int i=0;i<nr_unique_coeffs.size()-1;i++){
//         str += to_string(nr_unique_coeffs[i]) + ","; 
//       }
//       str += to_string(nr_unique_coeffs[nr_unique_coeffs.size()-1])+"\t";
//       csv_file << str;
// 
//       str = "";
//       for(int i=0;i<convergence.size()-1;i++){
//         str += to_string(convergence[i]) + ","; 
//       }
//       str += to_string(convergence[convergence.size()-1])+"\t";
//       csv_file << str;
// 
//       str = "";
//       for(int i=0;i<times.size()-1;i++){
//         str += to_string(times[i]) + ","; 
//       }
//       str += to_string(times[times.size()-1]);
//       csv_file << str + "\n";
//       csv_file.close();
//     }
// ============================================================================
  }

};







#endif