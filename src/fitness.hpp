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

  virtual ~Fitness() {};

  Mat X_train, X_val, X_batch, X_batch_opt;
  Vec y_train, y_val, y_batch, y_batch_opt;

  bool discount_size = false;


  virtual string name() {
    throw runtime_error("Not implemented");
  }

  virtual Fitness * clone() {
    throw runtime_error("Not implemented");
  }

  virtual vector<float> get_fitness_SO(Individual * n, Mat & X, Vec & y, bool change_fitness = true) {
    throw runtime_error("Not implemented");
  }

  virtual vector<float> get_fitness_MO(Individual * n, Mat & X, Vec & y, bool change_fitness = true){
    throw runtime_error("Not implemented");
  }
  
  // shorthand for training set
  vector<float> get_fitness_SO(Individual * n, Mat * X=NULL, Vec * y=NULL, bool change_fitness = true) {
    if (!X)
      X = & this->X_batch;
    if (!y)
      y = & this->y_batch;

    // update evaluations
    evaluations += 1;
    return get_fitness_SO(n, *X, *y, change_fitness);
  }

  // shorthand for training set
  vector<float> get_fitness_MO(Individual * n, Mat * X=NULL, Vec * y=NULL, bool change_fitness = true) {
      if (!X)
          X = & this->X_batch;
      if (!y)
          y = & this->y_batch;

      // update evaluations
      evaluations += 1;
    return get_fitness_MO(n, *X, *y, change_fitness);
  }

  void _set_X(Mat & X, string type="train") {
    if (type == "train") {
        X_train = X;
    }
    else if (type=="val") {

        X_val = X;
    }
    else{
      throw runtime_error("Unrecognized X type "+type);
      }
  }

  void _set_y(Vec & y, string type="train") {
    if (type == "train") {
        y_train = y;
    }
    else if (type=="val") {
        y_val = y;
    }
    else {
        throw runtime_error("Unrecognized y type " + type);
    }
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

struct MSEFitness : Fitness {

  string name() override {
    return "mse";
  }

  Fitness * clone() override {
    return new MSEFitness();
  }

  vector<float> get_fitness_SO(Individual * n, Mat & X, Vec & y, bool change_fitness=true) override {
    vector<float> fitnessses;
    fitnessses.reserve(1);
    Vec out = n->get_output(X);
    
    float fitness = (y-out).square().mean();

    if (isnan(fitness) || fitness < 0) // the latter can happen due to float overflow
      fitness = INF;

    if(change_fitness) {
        n->fitness[0] = fitness;
    }
    fitnessses.push_back(fitness);
    return fitnessses;
  }

  vector<float> get_fitness_MO(Individual * n, Mat & X, Vec & y, bool change_fitness=true) override {

    Vec out = n->get_output(X);

    float fitness = (y-out).square().mean();

    if (isnan(fitness) || fitness < 0) // the latter can happen due to float overflow
        fitness = INF;

    if(change_fitness) {
        n->fitness[0] = fitness;

        n->fitness[1] = n->get_num_nodes(true, discount_size);
    }

    return n->fitness;
  }

};



struct LSMSEFitness : Fitness {

  string name() override {
    return "lsmse";
  }

  Fitness * clone() override {
    return new LSMSEFitness();
  }

  vector<float> get_fitness_SO(Individual * n, Mat & X, Vec & y, bool change_fitness) override {
    Vec out = n->get_output(X);
    pair<float,float> intc_slope = linear_scaling_coeffs(y, out);


    if (intc_slope.second == 0){
        out = intc_slope.first + out;
    }
    else{
        out = intc_slope.first + intc_slope.second*out;
    }

    float fitness = (y-out).square().mean();
    if (isnan(fitness) || fitness < 0 || isinf(fitness)) // the latter can happen due to float overflow
      fitness = INF;

    if(change_fitness) {
        n->fitness[0] = fitness;
    }

    return n->fitness;
  }

  vector<float> get_fitness_MO(Individual * n, Mat & X, Vec & y,bool change_fitness) override {
        Vec out = n->get_output(X);
        pair<float,float> intc_slope = linear_scaling_coeffs(y, out);


        if (intc_slope.second == 0){
            out = intc_slope.first + out;
        }
        else{
            out = intc_slope.first + intc_slope.second*out;
        }


        float fitness = (y-out).square().mean();
        if (isnan(fitness) || fitness < 0 || isinf(fitness)) {
            fitness = INF;
        }

       if(change_fitness) {
        n->fitness[0] = fitness;
        n->fitness[1] = n->get_num_nodes(true, discount_size);
      }

        return n->fitness;
    }

};










#endif