#ifndef EVOLUTION_H
#define EVOLUTION_H

#include <Python.h>
#include <unordered_map>

#include "util.hpp"
#include "variation.hpp"
#include "selection.hpp"
#include "fos.hpp"
#include "globals.hpp"
#include "individual.hpp"
#include "rng.hpp"
#include <limits>

#include <iostream>
#include <fstream>

using namespace std;

struct Evolution {
  
  vector<Individual*> population;
  vector<vector<FOSBuilder *>> fbs;
  int gen_number = 0;
  int pop_size = 0;

  Evolution(int pop_size) {
    this->pop_size = pop_size;
    // For each MO cluster make for each multi-tree a fosbuilder. Fosbuilder cannot be shared due to initial bias
    if(g::MO_mode){
        fbs.reserve(g::n_clusters);
        for(int i = 0; i<g::n_clusters; i++){
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

    if(g::use_adf) {
        int size_terminals = g::terminals.size();

        for(int i =0;i<int(1);i++) {
            g::terminals.push_back(new AnyOp(0));
            g::terminals.push_back(new AnyOp(1));
        }
    }

    for(int i = 0; i<g::nr_multi_trees - 1;i++){
        if(g::use_aro) {
            g::terminals.push_back(new OutputTree(i));
        }
        if(g::use_adf) {
            g::functions.push_back(new FunctionTree(i));
        }
    }

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
        g::fit_func->get_fitness_SO(individual);


        population.push_back(individual);

    }

//      Individual * ind = new Individual();
//
//      Node * sinf = new Node(new Sin());
//      Node * pf = new Node(new Add());
//      Node * any0 = new Node(new AnyOp(0));
//      Node * any1 = new Node(new AnyOp(1));
//
//      Node * pa0 = new Node(new Add());
//      Node * pa1 = new Node(new Add());
//      Node * pa2 = new Node(new Add());
//      Node * pa3 = new Node(new Add());
//
//      Node * f11 = new Node(new Feat(0));
//      Node * f12 = new Node(new Feat(0));
//      Node * f13 = new Node(new Feat(0));
//      Node * f14 = new Node(new Feat(0));
//      Node * f15 = new Node(new Feat(0));
//      Node * f16 = new Node(new Feat(0));
//      Node * f17 = new Node(new Feat(0));
//      Node * f18 = new Node(new Feat(0));
//
//      pa0->append(f11);
//      pa0->append(f12);
//      pa1->append(f13);
//      pa1->append(f14);
//      pa2->append(f15);
//      pa2->append(f16);
//      pa3->append(f17);
//      pa3->append(f18);
//
//      any0->append(pa0);
//      any0->append(pa1);
//      any1->append(pa2);
//      any1->append(pa3);
//
//      pf->append(any0);
//      pf->append(any1);
//      sinf->append(pf);
//
//
//      Node * psin0 = new Node(new Add());
//      Node * psin1 = new Node(new Add());
//      Node * psin2 = new Node(new Add());
//
//      Node * psin3 = new Node(new Add());
//      Node * psin4 = new Node(new Add());
//      Node * psin5 = new Node(new Add());
//      Node * psin6 = new Node(new Add());
//
//      Node * s11 = new Node(new Feat(0));
//      Node * s12 = new Node(new Feat(0));
//      Node * s13 = new Node(new Feat(0));
//      Node * s14 = new Node(new Feat(0));
//      Node * s15 = new Node(new Feat(0));
//      Node * s16 = new Node(new Feat(0));
//      Node * s17 = new Node(new Feat(0));
//      Node * s18 = new Node(new Feat(0));
//
//      psin3->append(s11);
//      psin3->append(s12);
//      psin4->append(s13);
//      psin4->append(s14);
//      psin5->append(s15);
//      psin5->append(s16);
//      psin6->append(s17);
//      psin6->append(s18);
//
//      psin1->append(psin3);
//      psin1->append(psin4);
//      psin2->append(psin5);
//      psin2->append(psin6);
//
//      psin0->append(psin1);
//      psin0->append(psin2);
//
//
//
//      sinf->append(psin0);
//
//
//
//      ind->trees.push_back(sinf);
//
//
//      Node * p0 = new Node(new Add());
//
//      Node * p1 = new Node(new Add());
//      Node * p2 = new Node(new Add());
//
//      Node * p3 = new Node(new Add());
//      Node * p4 = new Node(new Add());
//      Node * p5 = new Node(new Add());
//      Node * p6 = new Node(new Add());
//
//      Node * ft1 = new Node(new FunctionTree(0));
//      Node * ft2 = new Node(new FunctionTree(0));
//      Node * ft3 = new Node(new FunctionTree(0));
//      Node * ft4 = new Node(new FunctionTree(0));
//      Node * ft5 = new Node(new FunctionTree(0));
//      Node * ft6 = new Node(new FunctionTree(0));
//      Node * ft7 = new Node(new FunctionTree(0));
//      Node * ft8 = new Node(new FunctionTree(0));
//
//      Node * f01 = new Node(new Feat(0));
//      Node * f02 = new Node(new Feat(0));
//      Node * f03 = new Node(new Feat(0));
//      Node * f04 = new Node(new Feat(0));
//      Node * f05 = new Node(new Feat(0));
//      Node * f06 = new Node(new Feat(0));
//      Node * f07 = new Node(new Feat(0));
//      Node * f08 = new Node(new Feat(0));
//
//      Node * f1 = new Node(new Feat(1));
//      Node * f2 = new Node(new Feat(2));
//      Node * f3 = new Node(new Feat(3));
//      Node * f4 = new Node(new Feat(4));
//      Node * f5 = new Node(new Feat(5));
//      Node * f6 = new Node(new Feat(6));
//      Node * f7 = new Node(new Feat(7));
//      Node * f8 = new Node(new Feat(7));
//
//      ft1->append(f01);
//      ft1->append(f1);
//      ft2->append(f02);
//      ft2->append(f2);
//      ft3->append(f03);
//      ft3->append(f3);
//      ft4->append(f04);
//      ft4->append(f4);
//      ft5->append(f05);
//      ft5->append(f5);
//      ft6->append(f06);
//      ft6->append(f6);
//      ft7->append(f07);
//      ft7->append(f7);
//      ft8->append(f08);
//      ft8->append(f8);
//
//      p3->append(ft1);
//      p3->append(ft2);
//      p4->append(ft3);
//      p4->append(ft4);
//      p5->append(ft5);
//      p5->append(ft6);
//      p6->append(ft7);
//      p6->append(ft8);
//
//      p1->append(p3);
//      p1->append(p4);
//      p2->append(p5);
//      p2->append(p6);
//
//      p0->append(p1);
//      p0->append(p2);
//
//      ind->trees.push_back(p0);
//
//      g::fit_func->get_fitness_SO(ind);
//      g::fit_func->get_fitness_MO(ind);
//
//      ind->clone();
//      population[0] = ind->clone();
//
//      print(ind->fitness[0]);












      g::ea->initMOArchive(population);
    g::ea->initSOArchive(population);

  }

    pair<pair<vector<vector<Individual *>>, vector<vector<Individual *>>>, vector<int>> single_cluster(vector<Individual *> &population){
        vector<vector<Individual *>> clustered_population = vector<vector<Individual *>>(1,vector<Individual *>());
        clustered_population[0] = population;
        vector<int> clusternr;
        if(g::MO_mode) {
            clusternr = vector<int>(1, g::nr_objs + 1);
        }
        else{
            clusternr = vector<int>(1, 0);
        }

        return make_pair(make_pair(clustered_population, clustered_population), clusternr);
  }

    pair<pair<vector<vector<Individual *>>, vector<vector<Individual *>>>, vector<int>>
    balanced_K_2_leader_means(vector<Individual *> &population) {
        int k = g::n_clusters;

        int nr_objs = g::nr_objs;

        // If 1 cluster then return the whole population in one cluster, the population as donor population, and all clusternr 3 (MO) or 0 in SO mode
        if(k<2 || !g::MO_mode){
            return single_cluster(population);
        }

        int pop_size = population.size();

        // keep track of possible leaders
        vector<int> range(pop_size);
        iota(range.begin(), range.end(), 0);

        // all solutions are still possible as leaders
        unordered_set<int> remaining_solutions(range.begin(), range.end());

        // get objective value data and normalise
        Mat norm_data(nr_objs, pop_size);
        for(int i=0; i<remaining_solutions.size(); i++){
            for(int j=0; j<nr_objs; j++){
                norm_data(j,i) = population[i]->fitness[j];
            }
        }

        vector<vector<Individual*>> top_clusters(nr_objs);

        auto rand_obj_perm_extrema = Rng::rand_perm(nr_objs);
        // Make two clusters
        for(int random_obj = 0; random_obj<nr_objs; random_obj++){
            rand_obj_perm_extrema[random_obj] = random_obj;
        //for(auto random_obj: rand_obj_perm_extrema) {
            vector<Individual*> extreme_cluster;
            vector<int> myVector(remaining_solutions.begin(), remaining_solutions.end());
            sort(myVector.begin(), myVector.end(), [&](int i, int j) {
                return population[i]->fitness[random_obj] < population[j]->fitness[random_obj];
            });

            for(int i = 0; i<floor(pop_size/k); i++){
                extreme_cluster.push_back(population[myVector[i]]);
                // Remove from remaining solutions
                remaining_solutions.erase(myVector[i]);
            }
            top_clusters[random_obj] = extreme_cluster;
        }



        for(int i=0; i<nr_objs; i++){
            float min;
            bool min_initialised = false;
            float max;
            bool max_initialised = false;
            // get min and max for objective
            for(int x: range){
                float value = norm_data(i, x);
                if ((!min_initialised || value < min) && !isinf(value)) {
                    min = value;
                    min_initialised = true;
                }
                if ((!max_initialised || value > max) && !isinf(value)) {
                    max = value;
                    max_initialised = true;
                }
            }
            //normalize each value in population for that objective
            for (int j = 0; j < pop_size; j++) {
                if(isinf(norm_data(i, j)) && norm_data(i, j)<0){
                    norm_data(i, j) = min;
                }
                if(isinf(norm_data(i, j))){
                    norm_data(i, j) = max;
                }
                norm_data(i, j) = (norm_data(i, j) - min) / ((max - min)+0.0000000001);
            }
        }

        // get first leaders based on smallest objective value
        int initialised_k = 0;
        vector<int> idx_leaders;

        auto rand_obj_perm = Rng::rand_perm(nr_objs);

        int idx_min = *remaining_solutions.begin();
        for (int x: remaining_solutions) {
            if (norm_data(rand_obj_perm[0], idx_min) > norm_data(rand_obj_perm[0], x)) {
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

        while(initialised_k<k-nr_objs && !remaining_solutions.empty()){
            // select leader with the furthest distance to other leaders
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
        // initialize matrix for centers of the clusters (start with location of the leaders)
        Mat cluster_centers = Mat::Zero(nr_objs, idx_leaders.size());
        for(int i =0; i<idx_leaders.size(); i++){
            cluster_centers.col(i) = norm_data.col(idx_leaders[i]);
        }

        //#unordered_set<int> new_remaining_solutions(range.begin(), range.end());
        unordered_set<int> new_remaining_solutions = remaining_solutions;


        while(new_remaining_solutions.size()>0){
            auto random_cluster_order = Rng::rand_perm(k-nr_objs);
            for(int j = 0; j<k-nr_objs; j++){
                int cluster_nr = random_cluster_order[j];

                // pick closest idx
                int closest_idx;
                bool lowest_dist_init = false;
                float lowest_dist;

                for(auto i: new_remaining_solutions){
                    // check the center that is closest for this indivdual
                    if (!lowest_dist_init ||
                        lowest_dist > (norm_data.col(i) - cluster_centers.col(cluster_nr)).square().mean()) {
                        lowest_dist = (norm_data.col(i) - cluster_centers.col(cluster_nr)).square().mean();
                        closest_idx = i;
                        lowest_dist_init = true;
                    }
                }
                //assign
                clustertags_kmeans[closest_idx]= cluster_nr;

                new_remaining_solutions.erase(closest_idx);
            }
        }

        vector<vector<int>> clustertags_equal = vector<vector<int>>(pop_size,vector<int>());
        // store the solutions of each cluster in a separate vector for FOS in clustered_population
        vector<vector<Individual *>> clustered_population = vector<vector<Individual *>>(k,vector<Individual *>());
        vector<vector<Individual *>> clustered_population_equal = vector<vector<Individual *>>(k,vector<Individual *>());

        // assign to each cluster the closest donor_fraction*pop_size/cluster solutions
        for(int x=0; x<k-nr_objs; x++){
            Vec distances = Vec(pop_size);
            // for each individual
            for(int i = 0; i<pop_size; i++){
                distances[i] = (norm_data.col(i) - cluster_centers.col(x)).square().mean();
            }
            int donor_pop_size = ((g::donor_fraction*pop_size)/(k));

            for(int times=0; times<donor_pop_size; times++){
                clustered_population_equal[x].push_back(population[argmin(distances)]);

                clustertags_equal[argmin(distances)].push_back(x);

                distances[argmin(distances)] = std::numeric_limits<float>::infinity();
            }
        }

        vector<int> clusternr = vector<int>(k, nr_objs + 1);

        // if not yet assigned solution in equal size clustering, assign to closest. if multiple are assigned, assign to random of multiple center
        for (auto i: remaining_solutions) {
            clustered_population[clustertags_kmeans[i]].push_back(population[i]);
        }

        for (auto i: idx_leaders) {
            clustered_population[clustertags_kmeans[i]].push_back(population[i]);
        }

        for(int i=0; i<nr_objs; i++){
            clustered_population[clustered_population.size()-(nr_objs-i)] = top_clusters[i];
            clustered_population_equal[clustered_population.size()-(nr_objs-i)] = top_clusters[i];
            clusternr[clusternr.size()-(nr_objs-i)] = rand_obj_perm_extrema[i];
        }


        return make_pair(make_pair(clustered_population, clustered_population_equal), clusternr);
    }

  pair<pair<vector<vector<Individual *>>, vector<vector<Individual *>>>, vector<int>>
  balanced_K_leader_means(vector<Individual *> &population) {
      int k = g::n_clusters;
      int nr_objs = g::nr_objs;

      // If 1 cluster then return the whole population in one cluster, the population as donor population, and all clusternr 3 (MO) or 0 in SO mode
      if(k<2 || !g::MO_mode){
          return single_cluster(population);
      }

      int pop_size = population.size();

      // keep track of possible leaders
      vector<int> range(pop_size);
      iota(range.begin(), range.end(), 0);

      // all solutions are still possible as leaders
      unordered_set<int> remaining_solutions(range.begin(), range.end());

      // get objective value data and normalise

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
              if ((!min_initialised || value < min) && !isinf(value)) {
                  min = value;
                  min_initialised = true;
              }
              if ((!max_initialised || value > max) && !isinf(value)) {
                  max = value;
                  max_initialised = true;
              }
          }
          //normalize each value in population for that objective
          for (int j = 0; j < pop_size; j++) {
              if(isinf(norm_data(i, j)) && norm_data(i, j)<0){
                  norm_data(i, j) = min;
              }
              if(isinf(norm_data(i, j))){
                  norm_data(i, j) = max;
              }
              norm_data(i, j) = (norm_data(i, j) - min) / ((max - min)+0.0000000001);
          }
      }

      // get first leaders based on smallest objective value
      int initialised_k = 0;
      vector<int> idx_leaders;

      auto rand_obj_perm = Rng::rand_perm(nr_objs);
      // TODO: here we force the elite of each objective to be a leader
      for(auto random_obj: rand_obj_perm) {
          int idx_min = *remaining_solutions.begin();
          for (int x: remaining_solutions) {
              if (norm_data(random_obj, idx_min) > norm_data(random_obj, x)) {
                  idx_min = x;
              }
          }
          int first_leader = idx_min;
          idx_leaders.push_back(first_leader);
          remaining_solutions.erase(first_leader);
          initialised_k++;
      }

      // find other leaders
      // first get distance of all remaining possible leaders to first leaders
      Vec dists = Vec::Zero(pop_size);
      for(int x: remaining_solutions){
          dists(x) = (norm_data.col(x) - norm_data.col(idx_leaders[0])).square().mean();
      }

      while(initialised_k<k && !remaining_solutions.empty()){
          // select leader with the furthest distance to other leaders
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
      // initialize matrix for centers of the clusters (start with location of the leaders)
      Mat cluster_centers = Mat::Zero(nr_objs, idx_leaders.size());
      for(int i =0; i<idx_leaders.size(); i++){
          cluster_centers.col(i) = norm_data.col(idx_leaders[i]);
      }

      unordered_set<int> new_remaining_solutions(range.begin(), range.end());

      while(new_remaining_solutions.size()>0){
          auto random_cluster_order = Rng::rand_perm(k);
          for(int j = 0; j<k; j++){
              int cluster_nr = random_cluster_order[j];

              // pick closest idx
              int closest_idx;
              bool lowest_dist_init = false;
              float lowest_dist;

              for(auto i: new_remaining_solutions){
                  // check the center that is closest for this indivdual
                  if (!lowest_dist_init ||
                      lowest_dist > (norm_data.col(i) - cluster_centers.col(cluster_nr)).square().mean()) {
                      lowest_dist = (norm_data.col(i) - cluster_centers.col(cluster_nr)).square().mean();
                      closest_idx = i;
                      lowest_dist_init = true;
                  }
              }
              //assign
              clustertags_kmeans[closest_idx]= cluster_nr;

              new_remaining_solutions.erase(closest_idx);
          }
      }

      vector<vector<int>> clustertags_equal = vector<vector<int>>(pop_size,vector<int>());
      // store the solutions of each cluster in a separate vector for FOS in clustered_population
      vector<vector<Individual *>> clustered_population = vector<vector<Individual *>>(initialised_k,vector<Individual *>());
      vector<vector<Individual *>> clustered_population_equal = vector<vector<Individual *>>(initialised_k,vector<Individual *>());

      // assign to each cluster the closest donor_fraction*pop_size/cluster solutions
      for(int x=0; x<initialised_k; x++){
          Vec distances = Vec(pop_size);
          // for each individual
          for(int i = 0; i<pop_size; i++){
              distances[i] = (norm_data.col(i) - cluster_centers.col(x)).square().mean();
          }
          int donor_pop_size = ((g::donor_fraction*pop_size)/initialised_k);

          for(int times=0; times<donor_pop_size; times++){
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

      // if not yet assigned solution in equal size clustering, assign to closest. if multiple are assigned, assign to random of multiple center
      for (int i = 0; i < pop_size; i++) {
          clustered_population[clustertags_kmeans[i]].push_back(population[i]);
      }




      return make_pair(make_pair(clustered_population, clustered_population_equal), clusternr);
  }

  pair<pair<vector<vector<Individual *>>, vector<vector<Individual *>>>, vector<int>>
  K_leader_means(vector<Individual *> &population) {
      int k = g::n_clusters;
      int nr_objs = g::nr_objs;
      if(k<2 || !g::MO_mode){
          return single_cluster(population);
      }

      int pop_size = population.size();

      // keep track of possible leaders
      vector<int> range(pop_size);
      iota(range.begin(), range.end(), 0);

      // all solutions are still possible as leaders
      unordered_set<int> remaining_solutions(range.begin(), range.end());

      // get objective value data and normalise

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
              if ((!min_initialised || value < min) && !isinf(value)) {
                  min = value;
                  min_initialised = true;
              }
              if ((!max_initialised || value > max) && !isinf(value)) {
                  max = value;
                  max_initialised = true;
              }
          }
          //normalize each value in population for that objective
          for (int j = 0; j < pop_size; j++) {
              if(isinf(norm_data(i, j)) && norm_data(i, j)<0){
                  norm_data(i, j) = min;
              }
              if(isinf(norm_data(i, j))){
                  norm_data(i, j) = max;
              }
              norm_data(i, j) = (norm_data(i, j) - min) / ((max - min)+0.0000000001);
          }
      }

      // get first leaders based on smallest objective value
      int initialised_k = 0;
      vector<int> idx_leaders;

      auto rand_obj_perm = Rng::rand_perm(nr_objs);

      // TODO: here we force the elite of each objective to be a leader
      for(auto random_obj: rand_obj_perm) {
          int idx_min = *remaining_solutions.begin();
          for (int x: remaining_solutions) {
              if (norm_data(random_obj, idx_min) > norm_data(random_obj, x)) {
                  idx_min = x;
              }
          }
          int first_leader = idx_min;
          idx_leaders.push_back(first_leader);
          remaining_solutions.erase(first_leader);
          initialised_k++;
      }

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
          Mat cluster_centers_temp = Mat::Zero(nr_objs, idx_leaders.size());
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

      // assign to each cluster the closest 2*pop_size/cluster solutions
      for(int x=0; x<initialised_k; x++){
          Vec distances = Vec(pop_size);
          for(int i = 0; i<pop_size; i++){
              distances[i] = (norm_data.col(i) - cluster_centers.col(x)).square().mean();

          }
          int donor_pop_size = ((g::donor_fraction*pop_size)/initialised_k);

          for(int times=0; times<donor_pop_size; times++){
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


      // if not yet assigned solution in equal size clustering, assign to closest. if multiple are assigned, assign to random of multiple center
      for (int i = 0; i < pop_size; i++) {
          if(!clustertags_equal[i].empty()){
              int assign;
              if(clustertags_equal[i].size()==1){
                  assign = clustertags_equal[i][0];
              }
              else{
                  assign = clustertags_equal[i][Rng::randu() * clustertags_equal[i].size()];
              }
              clustered_population[assign].push_back(population[i]);
          }
          else{
              clustered_population[clustertags_kmeans[i]].push_back(population[i]);
          }
      }


      // clusterpopulation population split into clusters
      // clusterpopulation equals use for donors

      return make_pair(make_pair(clustered_population, clustered_population_equal), clusternr);
  }

  void prune_duplicates(){
      vector<Individual *> keep;
      keep.reserve(pop_size);
      keep.push_back(population[0]->clone());

      int duplicates = 0;
      vector<string> stris;

      for(int i=1; i<pop_size;i++){
          bool add = true;
          for(auto ind:keep){
              bool all_same = true;
              for(int j=0;j<g::nr_objs;j++){
                  if(ind->fitness[j]!=population[i]->fitness[j]){
                      all_same = false;
                      break;
                  }
              }
              if(all_same){
                  add = false;

                  duplicates++;
                  break;
              }

          }


          if(add) {
              Individual * k = population[i]->clone();
              keep.push_back(k);
          }
          else{
              stris.push_back(population[i]->human_repr());
              if(g::replacement_strategy=="mutate") {
                  Individual * indi = population[i]->clone();
                  mutate(indi);
                  g::fit_func->get_fitness_MO(indi);
                  keep.push_back(indi);
              }
              else if(g::replacement_strategy=="archive"){
                  Individual * indi = g::ea->ReturnCopyRandomMOMember();
                  keep.push_back(indi);
              }
              else if(g::replacement_strategy=="sample"){
                  Individual * indi = generate_individuals(g::max_depth, g::init_strategy, g::nr_multi_trees);
                  g::fit_func->get_fitness_MO(indi);
                  keep.push_back(indi);
              }
              else{
                  throw std::invalid_argument( "Invalid replacement strategy");
              }



          }
      }

      print("DUPLICATES: ", duplicates);

//      int remaining = pop_size-keep.size();
//      for(int i=0; i<remaining; i++){
//          //Individual * ind = generate_individuals(g::max_depth, g::init_strategy, g::nr_multi_trees);
//
//
//          Individual * ind = population;
//
//          mutate(ind);
//
//          g::fit_func->get_fitness_MO(ind);
//
//          //keep.push_back(ind);
//          //keep.push_back(g::ea->ReturnCopyRandomMOMember());
//      }

      for(int i=0; i<population.size(); i++){
          population[i]->clear();
          population[i] = nullptr;
      }


      // This should be needed
      population.erase(std::remove_if(population.begin(), population.end(), [](Individual *ind){return ind== nullptr;}), population.end());

      population = keep;
  }

  void gomea_generation_MO(int macro_generation){
      if(g::remove_duplicates){
          prune_duplicates();
      }


      vector<Individual*> offspring_population;

      int nr_objectives = g::nr_objs;
      int NIS_const = 1 + log10(g::pop_size);
      //int NIS_const = 1;

      // Make clusters
      // ((Clustered Population, clustered population equal), cluster number)

      pair<pair<vector<vector<Individual*>>, vector<vector<Individual*>>>, vector<int>> output;
      if(g::balanced){
          //output = balanced_K_leader_means(population);
          output = balanced_K_leader_means(population);
      }
      else if(g::k2){
          output = balanced_K_2_leader_means(population);
      }
      else{
          output = K_leader_means(population);
      }

      vector<vector<Individual *>> clustered_population = output.first.first;
      vector<vector<Individual *>> clustered_donor_population = output.first.second;
      vector<int> clusternr = output.second;



      // Per cluster, one FOS
      vector<vector<pair<vector<int>,int>>> FOSs;
      vector<vector<vector<Node *>>> clustered_donor_pop;
      for(int i = 0; i<g::n_clusters; i++){
          vector<pair<vector<int>,int>> cluster_fbs;
          vector<vector<Node*>> donor_pop;
          for(int j = 0; j<g::nr_multi_trees; j++){
              if(clustered_population[i].size()>0) {
                  vector<Node *> fos_pop;
                  fos_pop.reserve(clustered_population[i].size());
                  for (int x = 0; x < clustered_population[i].size(); x++) {
                      fos_pop.push_back(clustered_population[i][x]->trees[j]);
                  }

                  vector<vector<int>> fos = fbs[i][j]->build_linkage_tree(fos_pop, j);
                  for (auto fos_el: fos) {
                      cluster_fbs.push_back(make_pair(fos_el, j));
                  }
              }
              if(clustered_donor_population[i].size()>0){
                  vector<Node *> fos_donor_pop;
                  for(int x=0;x<clustered_donor_population[i].size();x++){
                      fos_donor_pop.push_back(clustered_donor_population[i][x]->trees[j]);
                  }
                  donor_pop.push_back(fos_donor_pop);
              }
          }
          clustered_donor_pop.push_back(donor_pop);
          FOSs.push_back(cluster_fbs);
      }


      vector<pair<int, int>> idx;
      for(int i=0;i<clustered_population.size();i++){
          for(int j=0;j<clustered_population[i].size();j++){
              idx.emplace_back(i, j);
              clustered_population[i][j]->clusterid = i;
          }
      }

      if(g::log_pop && macro_generation>0){
          ofstream csv_file;
          csv_file.open(g::csv_file_pop, ios::app);

          string str = "";


          for (auto ind: population) {
              str += to_string(ind->fitness[0]) + "," + to_string(ind->fitness[1]) + "," + to_string(ind->fitness[2]) + "," +
                     to_string(ind->clusterid) + "," + to_string(clusternr[ind->clusterid]) + "," +
                     to_string(clustered_population[ind->clusterid].size()) + ";";
          }
          str.pop_back();
          str += "\n";


          for(int clusterid=0; clusterid<g::n_clusters;clusterid++){
              for (auto ind: clustered_population[clusterid]) {
                  str += to_string(ind->fitness[0]) + "," + to_string(ind->fitness[1]) + "," + to_string(ind->fitness[2]) + "," +
                         to_string(clusterid) + "," + to_string(clusternr[clusterid]) + "," +
                         to_string(clustered_population[clusterid].size()) + ";";
              }
          }
          str.pop_back();
          str += "\n";

          for(int clusterid=0; clusterid<g::n_clusters;clusterid++){
              for (auto ind: clustered_donor_population[clusterid]) {
                  str += to_string(ind->fitness[0]) + "," + to_string(ind->fitness[1]) + "," + to_string(ind->fitness[2]) + "," +
                         to_string(clusterid) + "," + to_string(clusternr[clusterid]) + "," +
                         to_string(clustered_donor_population[clusterid].size()) + ";";
              }
          }
          str.pop_back();
          str += "\n";

          for (auto ind: g::ea->MO_archive) {
              str += to_string(ind->fitness[0]) + "," + to_string(ind->fitness[1]) +  "," + to_string(ind->fitness[2]) + "," +
                     to_string(-1) + "," + to_string(-1) + "," +
                     to_string(-1) + ";";
          }
          str.pop_back();
          str += "\n";

          str += to_string(g::ea->min_objs[0]) + "," + to_string(g::ea->min_objs[1]) +  "," + to_string(g::ea->min_objs[2]) + "," + to_string(g::ea->max_objs[0]) + "," + to_string(g::ea->max_objs[1])  +  "," + to_string(g::ea->max_objs[2]) + "," + to_string(g::ea->num_boxes) + "\n";

          csv_file << str;
          csv_file.close();
      }






      for(int x=0; x<idx.size(); x++){
          int &i = idx[x].first;
          int &j = idx[x].second;

          Individual *offspring;
          if(clustered_population[i].size()>1){
              offspring = efficient_gom_MO(clustered_population[i][j], clustered_donor_pop[i], FOSs[i], macro_generation, clusternr[i], clusternr[i] < nr_objectives,  NIS_const);
          }
          else{
              offspring = efficient_gom_MO(clustered_population[i][j], clustered_donor_pop[i], FOSs[i], macro_generation,clusternr[i], clusternr[i] < nr_objectives, NIS_const);
          }

          offspring_population.push_back(offspring);

          g::ea->updateSOArchive(offspring);
          g::ea->updateMOArchive(offspring);
      }






      // TODO: !!!!!!!!!!!!
      assert(offspring_population.size()==pop_size);

      for(int i=0; i<population.size(); i++){
          population[i]->clear();
          population[i] = nullptr;
      }

      population = offspring_population;

      // This should be needed
      population.erase(std::remove_if(population.begin(), population.end(), [](Individual *ind){return ind== nullptr;}), population.end());




      g::ea->update_minmax();

  }


};

#endif