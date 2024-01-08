#ifndef FITNESS_H
#define FITNESS_H

#include <Eigen/Dense>
#include "myeig.hpp"
#include "individual.hpp"
#include "node.hpp"
#include "util.hpp"
#include "rng.hpp"

using namespace myeig;

struct Fitness {
  int opt_evaluations = 0;
  int evaluations = 0;
  long node_evaluations = 0;

  virtual ~Fitness() {};

  Mat X_train, X_val, X_batch, X_batch_opt;
  Vec y_train, y_val, y_batch, y_batch_opt;


  virtual string name() {
    throw runtime_error("Not implemented");
  }

  virtual Fitness * clone() {
    throw runtime_error("Not implemented");
  }

  virtual float get_fitness(Individual * n, Mat & X, Vec & y) {
    throw runtime_error("Not implemented");
  }

  virtual vector<float> get_fitness_MO(Individual * n, Mat & X, Vec & y){
    throw runtime_error("Not implemented");
  }
  
  // shorthand for training set
  float get_fitness(Individual * n, Mat * X=NULL, Vec * y=NULL) {
    if (!X)
      X = & this->X_batch;
    if (!y)
      y = & this->y_batch;

    // update evaluations
    evaluations += 1;
    //node_evaluations += n->get_num_nodes(true); 
    // call specific implementation
    return get_fitness(n, *X, *y);
  }

  // shorthand for training set
  vector<float> get_fitness_MO(Individual * n, Mat * X=NULL, Vec * y=NULL) {
    vector<float> fitness;
    fitness.reserve(2);
    fitness.push_back(get_fitness(n, X, y));
    fitness.push_back(n->get_num_nodes(true));
    return fitness;
  }

  float get_fitness_opt(Individual * n, Mat * X=NULL, Vec * y=NULL) {
    
    if (!X)
      X = & this->X_batch_opt;
    if (!y)
      y = & this->y_batch_opt;

    // update evaluations
    //node_evaluations += n->get_num_nodes(true); 
    // call specific implementation
    return get_fitness(n, *X, *y);
  }

  Vec get_fitnesses(vector<Individual*> population, bool compute=true, Mat * X=NULL, Vec * y=NULL) {  
    Vec fitnesses(population.size());
    for(int i = 0; i < population.size(); i++) {
      if (compute)
        fitnesses[i] = get_fitness(population[i], X, y);
      else
        fitnesses[i] = population[i]->fitness[0];
    }
    return fitnesses;
  }

  Vec get_fitnesses_MO(vector<Individual*> population, bool compute=true, Mat * X=NULL, Vec * y=NULL) {  
    //TODO: hardcoded size
    Vec fitnesses(population.size(), 2);
    for(int i = 0; i < population.size(); i++) {
      if(compute){
        fitnesses[i,0] = get_fitness(population[i], X, y);
      }
      else{
        fitnesses[i,0] = population[i]->fitness[0];
      }
      fitnesses[i,1] = population[i]->get_num_nodes(true);
    }
    return fitnesses;
  }


  void _set_X(Mat & X, string type="train") {
    if (type == "train")
      X_train = X;
    else if (type=="val")
      X_val = X;
    else
      throw runtime_error("Unrecognized X type "+type);
  }

  void _set_y(Vec & y, string type="train") {
    if (type == "train")
      y_train = y;
    else if (type=="val")
      y_val = y;
    else
      throw runtime_error("Unrecognized y type "+type);
  }

  void set_Xy(Mat & X, Vec & y, string type="train") {
    _set_X(X, type);
    _set_y(y, type);
    update_batch(X.rows());
  }

  bool update_batch(int num_observations) {

    int n = X_train.rows();

    if (num_observations==n) {
      X_batch = X_train;
      y_batch = y_train;
      return false;
    }
    
    // else pick some random elements
    auto chosen = Rng::rand_perm(num_observations);
    this->X_batch = X_train(chosen, Eigen::all);
    this->y_batch = y_train(chosen);
    return true;
  }

  bool update_batch_opt(int num_observations) {

    int n = X_train.rows();

    if (num_observations==n) {
      X_batch_opt = X_train;
      y_batch_opt = y_train;
      return false;
    }
    
    // else pick some random elements
    auto chosen = Rng::rand_perm(num_observations);
    this->X_batch_opt = X_train(chosen, Eigen::all);
    this->y_batch_opt = y_train(chosen);
    return true;
  }
};

struct MAEFitness : Fitness {

  string name() override {
    return "mae";
  }
  
  Fitness * clone() override {
    return new MAEFitness();
  }

  float get_fitness(Individual * n, Mat & X, Vec & y) override {
    Vec out = n->get_output(X, n->trees);

    float fitness = (y - out).abs().mean();
    if (isnan(fitness) || fitness < 0) // the latter can happen due to float overflow
      fitness = INF;
    n->fitness[0] = fitness;

    for(auto trees:n->trees){
        n->fitness[0] = fitness;
    }

    return fitness;;
  }

};

struct MSEFitness : Fitness {

  string name() override {
    return "mse";
  }

  Fitness * clone() override {
    return new MSEFitness();
  }

  float get_fitness(Individual * n, Mat & X, Vec & y) override {

    Vec out = n->get_output(X, n->trees);
    
    float fitness = (y-out).square().mean();

    if (isnan(fitness) || fitness < 0) // the latter can happen due to float overflow
      fitness = INF;

    n->fitness[0] = fitness;

    return fitness;
  }

};



struct LSMSEFitness : Fitness {

  string name() override {
    return "lsmse";
  }

  Fitness * clone() override {
    return new LSMSEFitness();
  }

  float get_fitness(Individual * n, Mat & X, Vec & y) override {
    Vec out = n->get_output(X, n->trees);
    pair<float,float> intc_slope = linear_scaling_coeffs(y, out);
    
    if (intc_slope.second == 0){
        out = intc_slope.first + out;
    }
    else{
        out = intc_slope.first + intc_slope.second*out;
    }


    float fitness = (y-out).square().mean();
    if (isnan(fitness) || fitness < 0) // the latter can happen due to float overflow
      fitness = INF;
    n->fitness[0] = fitness;


    return fitness;
  }

};










#endif