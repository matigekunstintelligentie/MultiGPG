#ifndef VARIATION_H
#define VARIATION_H

#include "globals.hpp"
#include "node.hpp"
#include "individual.hpp"
#include "operator.hpp"
#include "util.hpp"
#include "selection.hpp"
#include "fos.hpp"
#include "rng.hpp"

#include <dlib/optimization.h>
#include <dlib/global_optimization.h>
#include <vector>
#include <typeinfo>
#include <unsupported/Eigen/NonLinearOptimization>
//#include <unsupported/Eigen/LevenbergMarquardt>

using namespace std;

typedef dlib::matrix<double,0,1> column_vector;

Op * _sample_operator(vector<Op *> & operators, Vec & cumul_probs) {
  double r = Rng::randu();
  
  int j = 0;
  for(int i = 0; i<operators.size();i++){
    if(r <= (double) cumul_probs[i]){
      return operators[j]->clone(); 
      break;
    }
    j++;
  }
  
  Op * oppie = operators[operators.size() - 1]->clone();  
  return oppie;
}

Op * _sample_function() {
  return _sample_operator(g::functions, g::cumul_fset_probs);
}

Op * _sample_terminal(int mt, vector<Node *> &trees) {
    auto first = g::terminals.begin();
    auto last = g::terminals.end()-(g::nr_multi_trees-mt-1);
    vector<Op *> AB(first, last);
     
     Vec cumul_tset_probs(AB.size());

     cumul_tset_probs[0] = 1./float(AB.size());
     for(int i = 1; i<AB.size(); i++){
         cumul_tset_probs[i] = cumul_tset_probs[i-1] + 1./float(AB.size());
//print(to_string(cumul_tset_probs[i]), " ", to_string(g::cumul_tset_probs[i]), " ", AB[i]->sym(), " ", g::terminals[i]->sym());
     }

     return _sample_operator(AB, cumul_tset_probs);

    //return _sample_operator(g::terminals, g::cumul_tset_probs);
}

Node * _grow_tree_recursive(int max_arity, int max_depth_left, int actual_depth_left, int curr_depth, int mt, vector<Node *> &trees, float terminal_prob=.25) {
  Node * n = NULL;
  if (max_depth_left > 0) {
    if (actual_depth_left > 0 && Rng::randu() < 1.0-terminal_prob) {
      n = new Node(_sample_function());
    } else {
    
      n = new Node(_sample_terminal(mt, trees));
    }

    for (int i = 0; i < max_arity; i++) {
      Node * c = _grow_tree_recursive(max_arity,
        max_depth_left - 1, actual_depth_left - 1, curr_depth + 1, mt, trees, terminal_prob);
      n->append(c);
    }
  } else {
    n = new Node(_sample_terminal(mt, trees));
  }
  assert(n != NULL);

  return n;
}


Node * generate_tree(int max_depth, int mt, vector<Node *> &trees, string init_type="hh") {

  int max_arity = 0;
  for(Op * op : g::functions) {
    int op_arity = op->arity();
    if (op_arity > max_arity)
      max_arity = op_arity;
  }

  Node * tree = NULL;
  int actual_depth = max_depth;

  if (init_type == "rhh" || init_type == "hh") {
    if (init_type == "rhh")
      actual_depth = Rng::randu() * max_depth;
    
    bool is_full = Rng::randu() < .5;

    if (is_full)
      tree = _grow_tree_recursive(max_arity, max_depth, actual_depth, -1, mt, trees, 0.0);
    else{
      tree = _grow_tree_recursive(max_arity, max_depth, actual_depth, -1, mt, trees);
    }
  } else {
    throw runtime_error("Unrecognized init_type "+init_type);
  }
  
  if(g::add_addition_multiplication){
      Node * add_n, * mul_n, * slope_n, * interc_n;
      // mul_n = new Node(_sample_function());
      mul_n = new Node(new Mul());
      slope_n = new Node(new Const());
      // add_n = new Node(_sample_function());
      add_n = new Node(new Add());
      interc_n = new Node(new Const());
      mul_n->append(slope_n);
      mul_n->append(tree);
      add_n->append(interc_n);
      add_n->append(mul_n);
    
      // bring fitness info to new root
      add_n->fitness = tree->fitness;
      assert(add_n);
      return add_n;
  }
  else if(g::add_any){
      Node * add_n, * mul_n, * slope_n, * interc_n;
      mul_n = new Node(_sample_function());
      slope_n = new Node(new Const());
      add_n = new Node(_sample_function());
      interc_n = new Node(new Const());
      mul_n->append(slope_n);
      mul_n->append(tree);
      add_n->append(interc_n);
      add_n->append(mul_n);
    
      // bring fitness info to new root
      add_n->fitness = tree->fitness;
      assert(add_n);
      return add_n;
  }
  else{
    assert(tree);
    return tree;
  }
}


Individual * generate_individuals(int max_depth, string init_strategy, int nr_multi_trees){
    Individual * individual = new Individual();
    individual->trees.reserve(nr_multi_trees);
    for(int mt=0;mt<nr_multi_trees;mt++){
        individual->trees.push_back(generate_tree(max_depth, mt, individual->trees, init_strategy));
    }
    return individual;
}

Node * append_linear_scaling(Node * tree, vector<Node*> & trees) {
    // compute intercept and scaling coefficients, append them to the root
    Node * add_n, * mul_n, * slope_n, * interc_n;
  
    Vec p = tree->get_output(g::fit_func->X_train, trees);
  
    pair<float,float> intc_slope = linear_scaling_coeffs(g::fit_func->y_train, p);
    
    if (intc_slope.second == 0){
      add_n = new Node(new Add());
      interc_n = new Node(new Const(intc_slope.first));
      add_n->append(interc_n);
      add_n->append(tree);
      return add_n;
    }
  
    mul_n = new Node(new Mul());
    slope_n = new Node(new Const(intc_slope.second));
    add_n = new Node(new Add());
    interc_n = new Node(new Const(intc_slope.first));
    mul_n->append(slope_n);
    mul_n->append(tree);
    add_n->append(interc_n);
    add_n->append(mul_n);
  
    // bring fitness info to new root
    add_n->fitness = tree->fitness;
      
    return add_n;
return tree;
}

Node * coeff_opt_lm(Node * parent, vector<Node*> & trees,bool return_copy=true){
  Node * tree = parent;
  

   if(return_copy){
     tree = parent->clone();
   }

   vector<Node*> nodes = tree->subtree(true);

   vector<float> coeffsf;
   vector<Node*> coeff_ptrs;
   for(int i = 0; i < nodes.size(); i++) {
     if(nodes[i]->op->type()==OpType::otConst){
       coeffsf.push_back((float) ((Const*)nodes[i]->op)->c);
       coeff_ptrs.push_back(nodes[i]);
     }
   }

   int size = coeffsf.size() + 2;
   Eigen::VectorXf coeffsfv(size);
   for(int i=0; i<coeffsf.size(); i++) {
     coeffsfv(i) = coeffsf[i];
   }

   if(coeff_ptrs.size()>0){


     pair<float,float> intc_slope = make_pair(0.,1.);
     if(!g::use_mse_opt){
       Vec p = tree->get_output(g::fit_func->X_train, trees);
       intc_slope = linear_scaling_coeffs(g::fit_func->y_train, p);
     }

     coeffsfv(coeff_ptrs.size()) = intc_slope.first;
     coeffsfv(coeff_ptrs.size() + 1) = intc_slope.second;

     g::mse_func->update_batch_opt(g::batch_size_opt);

     if(g::use_optimiser){


               struct LMFunctor
               {
                   // 'm' pairs of (x, f(x))
                   vector<Node*> trees;
                   Mat X_train;
                   Vec y_train;
                   Node* tree;
                   std::vector<Node*> coeff_ptrs;
                   pair<float,float> intc_slope;

                   // Compute 'm' errors, one for each data point, for the given parameter values in 'x'
                   int operator()(const Eigen::VectorXf &x, Eigen::VectorXf &fvec) const
                   {
                       for(int i=0; i<coeff_ptrs.size(); i++){
                         ((Const*)coeff_ptrs[i]->op)->c = (float) x(i);
                       }
                       Mat X_train_tmp = X_train;
                       vector<Node*> trees_tmp = trees;
                       fvec = y_train - (x(coeff_ptrs.size()) + x(coeff_ptrs.size()+1) * tree->get_output(X_train_tmp, trees_tmp));
                       g::mse_func->opt_evaluations += 1;

                       return 0;
                   }

                   // Compute the jacobian of the errors
                   int df(const Eigen::VectorXf &x, Eigen::MatrixXf &fjac) const
                   {
                       g::jacobian_evals += 1;
                       // 'x' has dimensions n x 1
                       // It contains the current estimates for the parameters.

                       // 'fjac' has dimensions m x n
                       // It will contain the jacobian of the errors, calculated numerically in this case.


                        Mat X_train_tmp = X_train;

                        for(int i=0; i<coeff_ptrs.size(); i++){
                          ((Const*)coeff_ptrs[i]->op)->c = (float) x(i);
                        }

                        Mat Jacobian(X_train_tmp.rows(),coeff_ptrs.size() + 2);
                        Jacobian=Mat::Zero(X_train_tmp.rows(),coeff_ptrs.size() + 2);

                        for(int i=0; i<coeff_ptrs.size(); i++){
                            ((Const*)coeff_ptrs[i]->op)->d = static_cast<float>(1);

                            pair<Vec,Vec> output = tree->get_output_der(X_train_tmp);
                            g::mse_func->opt_evaluations += 1;
                            if(g::use_clip){
                              Jacobian.col(i) = (-output.second * x(coeff_ptrs.size()+1)).cwiseMin(-1).cwiseMax(1);
                            }
                            else{
                              Jacobian.col(i) = -output.second * x(coeff_ptrs.size()+1);
                            }


                            ((Const*)coeff_ptrs[i]->op)->d = static_cast<float>(0);
                        }

                        vector<Node*> trees_tmp = trees;
                        Vec output = tree->get_output(g::mse_func->X_batch_opt, trees_tmp);
                        g::mse_func->opt_evaluations += 1;


                        // TODO: change .size() to .rows() in gpg!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        if(!g::use_mse_opt){

                            Jacobian.col(coeff_ptrs.size()) = Vec::Zero(X_train_tmp.rows()) + 1.;
                            Jacobian.col(coeff_ptrs.size() + 1) = -output;
                        }
                        else{
                            Jacobian.col(coeff_ptrs.size()) = Vec::Zero(X_train_tmp.rows());
                            Jacobian.col(coeff_ptrs.size() + 1) = Vec::Zero(X_train_tmp.rows());
                        }

                        if(g::use_clip){
                         Jacobian.col(coeff_ptrs.size()) = Jacobian.col(coeff_ptrs.size()).cwiseMin(-1).cwiseMax(1);
                         Jacobian.col(coeff_ptrs.size() + 1) = Jacobian.col(coeff_ptrs.size() + 1).cwiseMin(-1).cwiseMax(1);
                        }


                        fjac = Jacobian.matrix();


                       return 0;
                   }

                   // Number of data points, i.e. values.
                   int m;

                   // Returns 'm', the number of values.
                   int values() const { return m; }

                   // The number of parameters, i.e. inputs.
                   int n;

                   // Returns 'n', the number of inputs.
                   int inputs() const { return n; }
               };

               LMFunctor functor;

               functor.trees = trees;
               functor.X_train = g::mse_func->X_batch_opt;
               functor.y_train = g::mse_func->y_batch_opt;
               functor.m = g::mse_func->X_batch_opt.rows();
               functor.n = g::mse_func->X_batch_opt.cols();
               functor.intc_slope = intc_slope;

               functor.tree = tree;
               functor.coeff_ptrs = coeff_ptrs;

               Eigen::LevenbergMarquardt<LMFunctor, float> lm(functor);
               lm.parameters.maxfev = g::lm_max_fev;
               //lm.parameters.factor = 10000000;
               lm.parameters.gtol = 0.;
               lm.parameters.ftol = 0.;
               lm.parameters.xtol = 0.;

               if(g::use_ftol){
                 lm.parameters.ftol = g::tol;
               }
               else{
                 lm.parameters.gtol = g::tol;
               }

               int status = lm.minimize(coeffsfv);

               for(int i=0; i<coeff_ptrs.size(); i++){
                 ((Const*)coeff_ptrs[i]->op)->c = coeffsfv(i);
               }
             }

   }
  return tree;
}

Node * coeff_mut(Node * parent, bool return_copy=true, vector<int> * changed_indices = NULL, vector<Op*> * backup_ops = NULL) {
  Node * tree = parent;

   if (return_copy) {
     tree = parent->clone();
   }

   if (g::cmut_prob > 0 && g::cmut_temp > 0) {
     // apply coeff mut to all nodes that are constants
     vector<Node*> nodes = tree->subtree();
     for(int i = 0; i < nodes.size(); i++) {
       Node * n = nodes[i];
       if (n->op->type() == OpType::otConst && Rng::randu() < g::cmut_prob) {

         float prev_c = ((Const*)n->op)->c;
         float std = g::cmut_temp*abs(prev_c);

         // cmut_eps = 0;
         if (std < g::cmut_eps){
           std = g::cmut_eps;
         }
         float mutated_c = prev_c + Rng::randn()*std;
         ((Const*)n->op)->c = mutated_c;
         if (changed_indices != NULL) {
           changed_indices->push_back(i);
           backup_ops->push_back(new Const(prev_c));
           if (find(changed_indices->begin(), changed_indices->end(), i) == changed_indices->end()) {
             changed_indices->push_back(i);
             backup_ops->push_back(new Const(prev_c));
           };
         }
       }
     }
   }

  return tree;
}

vector<int> _sample_crossover_mask(int num_nodes) {
  auto crossover_mask = Rng::rand_perm(num_nodes);
  int k = 1+sqrt(num_nodes)*abs(Rng::randn());
  k = min(k, num_nodes);
  crossover_mask.erase(crossover_mask.begin() + k, crossover_mask.end());
  assert(crossover_mask.size() == k);
  return crossover_mask;
}

Node * crossover(Node * parent, Node * donor) {
  Node * offspring = parent->clone();
  auto nodes = offspring->subtree();
  auto d_nodes = donor->subtree();

  // sample a crossover mask
  auto crossover_mask = _sample_crossover_mask(nodes.size());
  for(int i : crossover_mask) {
    delete nodes[i]->op;
    nodes[i]->op = d_nodes[i]->op->clone();
  }

  return offspring;
}

Node * mutation(Node * parent, vector<Op*> & functions, vector<Op*> & terminals, float prob_fun = 0.75) {
  Node * offspring = parent->clone();
// ============================================================================
//   auto nodes = offspring->subtree();
// 
//   // sample a crossover mask
//   auto crossover_mask = _sample_crossover_mask(nodes.size());
//   for(int i : crossover_mask) {
//     delete nodes[i]->op;
//     if (nodes[i]->children.size() > 0 && Rng::randu() < prob_fun) {
//       nodes[i]->op = _sample_function();
//     }
//     else {
//       nodes[i]->op = _sample_terminal();
//     }
//   }
// ============================================================================

  return offspring;
}
 


Node * efficient_gom(Individual * parent, int mt, vector<Node*> & population, vector<vector<int>> & fos, int macro_generations) {
//  Node * offspring = parent->trees[mt]->clone();
  Node * offspring = parent->trees[mt];

  float backup_fitness = parent->fitness;
  vector<Node*> offspring_nodes = offspring->subtree();



  auto random_fos_order = Rng::rand_perm(fos.size());

  bool ever_improved = false;

  for(int fos_idx = 0; fos_idx < fos.size(); fos_idx++) {
    
    auto crossover_mask = fos[random_fos_order[fos_idx]];
    bool change_is_meaningful = false;
    vector<Op*> backup_ops; backup_ops.reserve(crossover_mask.size());
    vector<int> effectively_changed_indices; effectively_changed_indices.reserve(crossover_mask.size());

    Node * donor = population[Rng::randi(population.size())];
    vector<Node*> donor_nodes = donor->subtree();

    for(int & idx : crossover_mask) {
      // check if swap is not necessary
      if (offspring_nodes[idx]->op->sym() == donor_nodes[idx]->op->sym()) {
        // might need to swap if the node is a constant that might be optimized
        if (g::cmut_prob <= 0 || g::cmut_temp <= 0 || donor_nodes[idx]->op->type() != OpType::otConst)
          continue;
      }

      // then execute the swap
      Op * replaced_op = offspring_nodes[idx]->op;
      offspring_nodes[idx]->op = donor_nodes[idx]->op->clone();
      backup_ops.push_back(replaced_op);
      effectively_changed_indices.push_back(idx);
    }



    // check if at least one change was meaningful
    for(int i : effectively_changed_indices) {
      Node * n = offspring_nodes[i];
      if (!n->is_intron()) {
        change_is_meaningful = true;
        break;
      }
    }

    // assume nothing changed
    float new_fitness = backup_fitness;
    if (change_is_meaningful) {
      // gotta recompute

      //parent->trees[mt] = offspring;
      new_fitness = g::fit_func->get_fitness(parent);
    }

    // check is not worse
    if (new_fitness > backup_fitness) {
      // undo
      for(int i = 0; i < effectively_changed_indices.size(); i++) {
        int changed_idx = effectively_changed_indices[i];
        Node * off_n = offspring_nodes[changed_idx];
        Op * back_op = backup_ops[i];
        delete off_n->op;
        off_n->op = back_op->clone();
        offspring->fitness = backup_fitness;
      }
      //parent->trees[mt] = offspring;
      parent->fitness = backup_fitness;
    } else if (new_fitness < backup_fitness) {
      // it improved

      backup_fitness = new_fitness;
      
      ever_improved = true;
    }

    // discard backup
    for(Op * op : backup_ops) {
      delete op;
    }
    
    if(g::cmut_prob>0.){

       effectively_changed_indices.clear();
       backup_ops.clear();

       // apply coeff mut

       coeff_mut(offspring, false, &effectively_changed_indices, &backup_ops);

       // check if at least one change was meaningful
       for(int i : effectively_changed_indices) {
         Node * n = offspring_nodes[i];
         if (!n->is_intron()) {
           change_is_meaningful = true;
           break;
         }
       }
       // assume nothing changed
       new_fitness = offspring->fitness;
       if (change_is_meaningful) {
         // gotta recompute
         new_fitness = g::fit_func->get_fitness(parent);
       }



       // check is not worse
       if (new_fitness > backup_fitness) {
         if(Rng::randu()<g::random_accept_p){
           // it didn't improve, but got accepted
           backup_fitness = new_fitness;
         }
         else{
           // undo
           for(int i = 0; i < effectively_changed_indices.size(); i++) {
             int changed_idx = effectively_changed_indices[i];
             Node * off_n = offspring_nodes[changed_idx];
             Op * back_op = backup_ops[i];
             delete off_n->op;
             off_n->op = back_op->clone();
             offspring->fitness = backup_fitness;
           }
         }
       } else if (new_fitness < backup_fitness) {
         // it improved
         backup_fitness = new_fitness;

         ever_improved = true;
       }

       // discard backup
       for(Op * op : backup_ops) {
         delete op;
       }
     }
   }




   if((macro_generations%g::opt_per_gen)==0 && g::use_optimiser && mt==(g::nr_multi_trees - 1)){

     offspring_nodes = offspring->subtree(parent->trees);
     bool coeff_found = false;
     for(int i = 0; i < offspring_nodes.size(); i++) {
       if(offspring_nodes[i]->op->type()==OpType::otConst && !offspring_nodes[i]->is_intron()){
         coeff_found = true;
         break;
       }
     }

     if(coeff_found){
       float fitness_before = offspring->fitness;

       Node * offspring_opt;
       if(g::optimiser_choice=="lm"){
         offspring_opt = coeff_opt_lm(offspring, parent->trees, true);
       }

//       //Possibly redundant
        parent->trees[mt] = offspring_opt;
        float fitness_after = g::fit_func->get_fitness(parent);

//
        if(fitness_before > fitness_after){
           offspring->clear();
           offspring = offspring_opt;
           ever_improved = true;
        }
        else{
            offspring_opt->clear();
        }
     }
   }
  

   // variant of forced improvement that is potentially less aggressive, & less expensive to carry out
    if(g::tournament_size > 1 && !ever_improved) {
      // make a tournament between tournament size - 1 candidates + offspring
      vector<Node*> tournament_candidates; tournament_candidates.reserve(g::tournament_size - 1);
      for(int i = 0; i < g::tournament_size - 1; i++) {
        tournament_candidates.push_back(population[Rng::randi(population.size())]);
      }
       tournament_candidates.push_back(offspring);

       Node * winner = tournament(tournament_candidates, g::tournament_size);
       offspring->clear();
       offspring = winner;
    }

  return offspring;
}

Individual * efficient_gom(Individual * og_parent, vector<Individual *> & indpopulation, vector<vector<vector<int>>> & multi_fos, int macro_generations){
    Individual * parent = og_parent->clone();

    for(int mt=0;mt<g::nr_multi_trees;mt++){
        vector<Node*> population;
        population.reserve(indpopulation.size());
        for(int i =0; i<indpopulation.size();i++){
            population.push_back(indpopulation[i]->trees[mt]);
        }
//

        Node * new_tree = efficient_gom(parent, mt, population, multi_fos[mt], macro_generations);
//
        parent->trees[mt] = new_tree;
    }
    return parent;
}



#endif