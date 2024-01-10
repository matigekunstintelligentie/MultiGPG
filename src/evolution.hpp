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
#include "rng.hpp"
#include <limits>

struct Evolution {
  
  vector<Individual*> population;
  vector<vector<FOSBuilder *>> fbs;
  int gen_number = 0;
  int pop_size = 0;

  Evolution(int pop_size) {
    this->pop_size = pop_size;
    // For each MO cluster make for each multi-tree a fosbuilder. Fosbuilder cannot be shared due to initial bias
    if(g::MO_mode){
        fbs.reserve(7);
        for(int i = 0; i<7; i++){
            vector<FOSBuilder *> cluster_fbs;
            for(int j = 0; j<g::nr_multi_trees; j++){
                cluster_fbs.push_back(new FOSBuilder());
            }
            fbs.push_back(cluster_fbs);
        }
    }
    else {
        fbs.reserve(1);
        vector<FOSBuilder *> cluster_fbs;
        for (int j = 0; j < g::nr_multi_trees; j++) {
            cluster_fbs.push_back(new FOSBuilder());
        }
        fbs.push_back(cluster_fbs);
    }
    init_pop();
  }

  ~Evolution() {
    clear_population(population);
    for(auto cluster_fbs : fbs){
        for(auto mt_fb: cluster_fbs){
            if(mt_fb){
                delete mt_fb;
            }
        }
    }
  }

  void clear_population(vector<Individual*> & population) {
    for(auto * individual : population) {
      individual->clear();
    }
    population.clear();
  }

  void init_pop() {
    unordered_set<string> already_generated;
    population.reserve(pop_size);
    int init_attempts = 0;
    while (population.size() < pop_size) {
      auto * individual = generate_individuals(g::max_depth, g::init_strategy, g::nr_multi_trees);

      string str_tree = individual->human_repr();
      if (init_attempts < g::max_init_attempts && already_generated.find(str_tree) != already_generated.end()) {
           individual->clear();
           init_attempts++;
           if (init_attempts == g::max_init_attempts) {
             print("[!] Warning: could not initialize a syntactically-unique population within ", init_attempts, " attempts");
           }
           continue;
         }
        already_generated.insert(str_tree);
        g::fit_func->get_fitness_MO(individual);
        population.push_back(individual);
    }
  } 

  void gomea_generation_SO(int macro_generation) {
    vector<vector<vector<int>>> foses;
    foses.reserve(g::nr_multi_trees);
    for(int i =0; i<g::nr_multi_trees;i++){
        vector<Node *> fos_pop;
        fos_pop.reserve(population.size());
        for(int j =0; j<population.size();j++){
            fos_pop.push_back(population[j]->trees[i]);
        }
        foses.push_back(fbs[0][i]->build_linkage_tree(fos_pop, i));
    }

    // perform GOM
    vector<Individual*> offspring_population;
    offspring_population.reserve(pop_size);

    for(int i = 0; i < pop_size; i++) {
      Individual * offspring = efficient_gom_SO(population[i], population, foses, macro_generation);
      //check_n_set_elite(offspring);
      offspring_population.push_back(offspring);
    }
    // replace parent with offspring population
    clear_population(population);

    population = offspring_population;

    ++gen_number;
  }

  pair<pair<vector<vector<Individual *>>, vector<vector<Individual *>>>, vector<int>>
  K_leader_means(vector<Individual *> &population) {
      int k = 7;
      int pop_size = population.size();

      // keep track of possible leaders
      vector<int> range(pop_size);
      iota(range.begin(), range.end(), 0);
      // all solutions are still possible as leaders
      unordered_set<int> remaining_solutions(range.begin(), range.end());

      // get objective value data and normalise
      int nr_objs = 2;
      Mat norm_data(nr_objs, pop_size);
      for(int i=0; i<pop_size; i++){
          for(int j=0; j<nr_objs; j++){
              norm_data(j,i) = population[i]->fitness[j];
          }
      }

      for(int i=0; i<nr_objs; i++){
          float min;
          bool min_initialised = false;
          float max;
          bool max_initialised = false;
          // get min and max for objective
          for(int x: remaining_solutions){
              float value = norm_data(i, x);
              if (!min_initialised || value < min) {
                  min = value;
                  min_initialised = true;
              }
              if (!min_initialised || value > max) {
                  max = value;
                  min_initialised = true;
              }
          }
          //normalize each value in population for that objective
          for (int j = 0; j < pop_size; j++) {
              norm_data(i, j) = (norm_data(i, j) - min) / ((max - min)+0.0000000001);
          }
      }
      // get first leaders based on smallest objective value
      int initialised_k = 0;
      vector<int> idx_leaders;
      // take a random objective
      int random_obj = round(Rng::randu()*(nr_objs-1));

      int idx_min = *remaining_solutions.begin();
      for(int x: remaining_solutions){
          if(norm_data(random_obj, idx_min)>norm_data(random_obj,x)){
              idx_min = x;
          }
      }
      int first_leader = idx_min;
      idx_leaders.push_back(first_leader);
      remaining_solutions.erase(first_leader);
      initialised_k++;

      // find other leaders
      // first get distance of all remaining possible leaders to first leaders
      Vec dists = Vec::Zero(pop_size);
      for(int x: remaining_solutions){
          dists(x) = (norm_data.col(x) - norm_data.col(idx_leaders[0])).square().mean();
      }

      while(initialised_k<k && !remaining_solutions.empty()){
          // select leader with longest distance to other leaders
          int new_leader = argmax(dists);
          idx_leaders.push_back(new_leader);
          initialised_k++;
          // minimum distance of new leader to other leaders now becomes 0 and the new leader is not a remaining possible leader
          dists[new_leader] = 0;
          remaining_solutions.erase(new_leader);
          // update distances of possible new leaders: if the new solution is closer to the individual then the one currently closest we update the distance
          for (size_t x: remaining_solutions) {
              double minimum = min((norm_data.col(x) - norm_data.col(new_leader)).square().mean(), dists[x]);
              dists[x] = minimum;
          }
      }

      // intialize cluster tags for k-means
      vector<int> clustertags_kmeans(pop_size);
      // intialize bool for whether there was a change in cluster assignments
      bool cluster_change = true;
      // initialize matrix for centers of the clusters (start with location of the leaders)
      Mat cluster_centers = Mat::Zero(nr_objs, idx_leaders.size());
      for(int i =0; i<idx_leaders.size(); i++){
          cluster_centers.col(i) = norm_data.col(idx_leaders[i]);
      }

      // k-means
      int iter = 0;
      while(cluster_change && iter<50){
          cluster_change = false;
          Mat cluster_centers_temp = Vec::Zero(nr_objs, idx_leaders.size());
          vector<int> counts = vector<int>(idx_leaders.size(), 0);

          for(int i = 0; i<pop_size; i++){
              bool lowest_dist_init = false;
              float lowest_dist;
              int index_min;
              // check the center that is closest for this indivdual
              for (size_t j = 0; j < idx_leaders.size(); j++) {
                  if (!lowest_dist_init ||
                      lowest_dist > (norm_data.col(i) - cluster_centers.col(j)).square().mean()) {
                      lowest_dist = (norm_data.col(i) - cluster_centers.col(j)).square().mean();
                      index_min = j;
                      lowest_dist_init = true;
                  }
              }
              // if it is different than before we update it
              if (index_min != clustertags_kmeans[i]) {
                  clustertags_kmeans[i] = index_min;
                  cluster_change = true;
              }
              // cluster_centers_temp are updated as well as the number of individuals belonging to a center, for calculating the new clustercenters later on
              cluster_centers_temp.col(clustertags_kmeans[i]) += norm_data.col(i);
              counts[clustertags_kmeans[i]] += 1;
          }
          for(int nr_lead=0; nr_lead<idx_leaders.size(); nr_lead++){
              // if the center doesn't have any individuals, reset to position leader
              if (counts[nr_lead] == 0) {
                  cluster_centers.col(nr_lead) = norm_data.col(idx_leaders[nr_lead]);
                  clustertags_kmeans[idx_leaders[nr_lead]] = nr_lead;
              }
              // else finalize new cluster center by calculating sum values of all individuals divided by the number of individuals of that cluster
              else{
                  for (size_t obj = 0; obj < nr_objs; obj++) {
                      cluster_centers(obj, nr_lead) = cluster_centers_temp(obj, nr_lead) / counts[nr_lead];
                  }
              }
          }
          iter++;
      }

      vector<vector<int>> clustertags_equal = vector<vector<int>>(pop_size,vector<int>());
      // store the solutions of each cluster in a separate vector for FOS in clustered_population
      vector<vector<Individual *>> clustered_population = vector<vector<Individual *>>(initialised_k,vector<Individual *>());
      vector<vector<Individual *>> clustered_population_equal = vector<vector<Individual *>>(initialised_k,vector<Individual *>());

      // assign to each cluster the closests 2*pop_size/cluster solutions
      for(int x=0; x<initialised_k; x++){
          Vec distances = Vec(pop_size);
          for(int i = 0; i<pop_size; i++){
              distances[i] = (norm_data.col(i) - cluster_centers.col(x)).square().mean();
          }
          for(int times=0; times<((2*pop_size)/initialised_k); times++){
              clustered_population_equal[x].push_back(population[argmin(distances)]);
              clustertags_equal[argmin(distances)].push_back(x);
              distances[argmin(distances)] = std::numeric_limits<float>::infinity();
          }
      }

      vector<int> clusternr = vector<int>(initialised_k, nr_objs + 1);

      for(int i=0; i<nr_objs; i++){
          int am = 0;
          float min_val = std::numeric_limits<float>::infinity();
          for(int j=0; j<cluster_centers.row(i).size();j++){
              if(cluster_centers.row(i)(j)<min_val){
                  min_val = cluster_centers.row(i)(j);
                  am = j;
              }
          }
          clusternr[am] = i;
      }

      // if not jet assigned solution in equal size clustering, assign to closest. if multiple are assigned, assign to random of multiple center
      for (size_t i = 0; i < pop_size; i++) {
          
      }

  }

//  void gomea_MO_generation(int macro_generation){
//      vector<Individual*> offspring_population;
//
//      int nr_objectives = 2;
//      int NIS = 1 + log10(g::pop_size);
//
//      // Make clusters
//      pair<pair<vector<vector<Individual*>>, vector<vector<Individual*>>>, vector<int>> output = K_leader_means(population);
//      vector<vector<Individual *>> clustered_population = output.first.first;
//      vector<vector<Individual *>> clustered_population_equal = output.first.second;
//      vector<int> clusternr = output.second;
//
//      vector<vector<vector<vector<int>>>> FOSs;
//      for(int i=0; i<clustered_population.size();i++){
//          FOSs.push_back(fb->build_linkage_tree(clustered_population_equal[i]));
//      }
//
//      vector<pair<int, int>> idx;
//      for(int i=0;i<clustered_population.size();i++){
//          for(int j=0;j<clustered_population.size();j++){
//              idx.emplace_back(i, j);
//          }
//      }
//
////      for(int x=0; x<idx.size(); x++){
////          int &i = idx[x].first;
////          int &j = idx[x].second;
////
////          Individual *offspring;
////          if(clustered_population.size()>1){
////              offspring = GOMMO();
////          }
////          else{
////              offspring = GOMMO();
////          }
////          offspring_population.push_back(offspring);
////      }
//
//      assert(offspring_population.size()==population.size());
//
//      for(int i=0; i<population.size(); i++){
//          population[i]->clear();
//      }
//
//      population = offspring_population;
//  }


  void run() {
    throw runtime_error("Not implemented, please use IMS (with max runs 1 if you want a single population)");
  }

};

#endif