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
#include "bfgs.h"
#include <vector>
#include <typeinfo>
#include <unsupported/Eigen/NonLinearOptimization>


using namespace std;

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
    return operators[operators.size() - 1]->clone();
}

Op * _sample_function(int mt, vector<Node *> &trees) {
    auto first = g::functions.begin();
    auto last = g::functions.end();
    if(g::use_adf) {
        last = g::functions.end() - (g::nr_multi_trees - mt - 1);
    }

    vector<Op *> AB(first, last);

    Vec cumul_fset_probs(AB.size());

    cumul_fset_probs[0] = 1./float(AB.size());
    for(int i = 1; i<AB.size(); i++){
        cumul_fset_probs[i] = cumul_fset_probs[i-1] + 1./float(AB.size());
    }

    return _sample_operator(AB, cumul_fset_probs);
}

Op * _sample_terminal(int mt, vector<Node *> &trees) {
    auto first = g::terminals.begin();
    auto last = g::terminals.end()-(g::nr_multi_trees-mt-1);
    vector<Op *> AB(first, last);
     
     Vec cumul_tset_probs(AB.size());

     cumul_tset_probs[0] = 1./float(AB.size());
     for(int i = 1; i<AB.size(); i++){
         cumul_tset_probs[i] = cumul_tset_probs[i-1] + 1./float(AB.size());
     }

     return _sample_operator(AB, cumul_tset_probs);
}

Node * _grow_tree_recursive(int max_arity, int max_depth_left, int actual_depth_left, int curr_depth, int mt, vector<Node *> &trees, float terminal_prob=.25) {
  Node * n = NULL;
  if (max_depth_left > 0) {
    if (actual_depth_left > 0 && Rng::randu() < 1.0-terminal_prob) {
      n = new Node(_sample_function(mt, trees));
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

    if (is_full) {
        tree = _grow_tree_recursive(max_arity, max_depth, actual_depth, -1, mt, trees, 0.0);
    }
    else{
      tree = _grow_tree_recursive(max_arity, max_depth, actual_depth, -1, mt, trees);
    }
  } else {
    throw runtime_error("Unrecognized init_type "+init_type);
  }
  


    assert(tree);
    return tree;

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
}

Individual * coeff_opt_lm(Individual * parent, bool return_copy=true){
    Individual * tree = parent;


    if(return_copy){
        tree = parent->clone();
    }

    vector<Node*> nodes = tree->subtree(false);

    // Get list of pointers to coeffs and their values
    vector<float> coeffsf;
    vector<Node*> coeff_ptrs;
    for(int i = 0; i < nodes.size(); i++) {
      if(nodes[i]->op->type()==OpType::otConst){

          if(std::find(coeff_ptrs.begin(), coeff_ptrs.end(), nodes[i]) == coeff_ptrs.end()) {
              coeff_ptrs.push_back(nodes[i]);
              coeffsf.push_back((float) ((Const*)nodes[i]->op)->c);
          }

      }
    }

   // Fill vector with coeffs with 2 extra for linear scaling
   int size = coeffsf.size() + 2;
   Eigen::VectorXf coeffsfv(size);
   for(int i=0; i<coeffsf.size(); i++) {
     coeffsfv(i) = coeffsf[i];
   }

   if(coeff_ptrs.size()>0){

     pair<float,float> intc_slope = make_pair(0.,1.);
     if(!g::use_mse_opt){
       Vec p = tree->get_output(g::fit_func->X_train);
       intc_slope = linear_scaling_coeffs(g::fit_func->y_train, p);
     }

     coeffsfv(coeff_ptrs.size()) = intc_slope.first;
     coeffsfv(coeff_ptrs.size() + 1) = intc_slope.second;

     g::mse_func->update_batch_opt(g::batch_size_opt);

     struct LMFunctor{
       // 'm' pairs of (x, f(x))
       Mat X_train;
       Vec y_train;
       Individual* tree;
       std::vector<Node*> coeff_ptrs;
       pair<float,float> intc_slope;

       // Compute 'm' errors, one for each data point, for the given parameter values in 'x'
       int operator()(const Eigen::VectorXf &x, Eigen::VectorXf &fvec) const{
           for(int i=0; i<coeff_ptrs.size(); i++){
               ((Const*)coeff_ptrs[i]->op)->c = (float) x(i);
           }

           fvec = y_train - (x(coeff_ptrs.size()) + x(coeff_ptrs.size()+1) * tree->get_output(X_train));

           g::mse_func->opt_evaluations += 1;

           return 0;
       }

       // Compute the jacobian of the errors
       int df(const Eigen::VectorXf &x, Eigen::MatrixXf &fjac) const{
           // 'x' has dimensions n x 1
           // It contains the current estimates for the parameters.

           // 'fjac' has dimensions m x n
           // It will contain the jacobian of the errors, calculated numerically in this case.

           // Update the coefficients in the tree
           for(int i=0; i<coeff_ptrs.size(); i++){
               ((Const*)coeff_ptrs[i]->op)->c = (float) x(i);
           }

           Mat Jacobian(X_train.rows(),coeff_ptrs.size() + 2);
           Jacobian=Mat::Zero(X_train.rows(),coeff_ptrs.size() + 2);

           for(int i=0; i<coeff_ptrs.size(); i++){
               // Mark the operator as being differentiable
               ((Const*)coeff_ptrs[i]->op)->d = static_cast<float>(1);

               pair<Vec,Vec> output = tree->get_output_der(X_train);

               g::mse_func->opt_evaluations += 1;
               if(g::use_clip){
                   Jacobian.col(i) = (-output.second * x(coeff_ptrs.size()+1)).cwiseMax(-1).cwiseMin(1);
               }
               else{
                   Jacobian.col(i) = -output.second * x(coeff_ptrs.size()+1);
               }
               ((Const*)coeff_ptrs[i]->op)->d = static_cast<float>(0);
           }

           Vec output = tree->get_output(g::mse_func->X_batch_opt);
           g::mse_func->opt_evaluations += 1;

           if(!g::use_mse_opt){
                    Jacobian.col(coeff_ptrs.size()) = Vec::Zero(X_train.rows()) + 1.;
                    Jacobian.col(coeff_ptrs.size() + 1) = -output;
           }
           else{
                    Jacobian.col(coeff_ptrs.size()) = Vec::Zero(X_train.rows());
                    Jacobian.col(coeff_ptrs.size() + 1) = Vec::Zero(X_train.rows());
           }

           if(g::use_clip){
               Jacobian.col(coeff_ptrs.size()) = Jacobian.col(coeff_ptrs.size()).cwiseMax(-1).cwiseMin(1);
               Jacobian.col(coeff_ptrs.size() + 1) = Jacobian.col(coeff_ptrs.size() + 1).cwiseMax(-1).cwiseMin(1);
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

Individual * efficient_gom(Individual * og_parent, vector<vector<Node*>> & mt_population, vector<Individual*> & population, vector<pair<vector<int>, int>> & fos, int macro_generations) {
    Individual * parent = og_parent->clone();
    auto random_fos_order = Rng::rand_perm(fos.size());

    bool ever_improved = false;
    vector<float> backup_fitness = parent->fitness;

    for(int fos_idx = 0; fos_idx < fos.size(); fos_idx++) {
        int mt = fos[random_fos_order[fos_idx]].second;
        auto crossover_mask = fos[random_fos_order[fos_idx]].first;
        Node * offspring = parent->trees[mt];
        vector<Node*> offspring_nodes = offspring->subtree();

        bool change_is_meaningful = false;
        vector<Op*> backup_ops; backup_ops.reserve(crossover_mask.size());
        vector<int> effectively_changed_indices; effectively_changed_indices.reserve(crossover_mask.size());

        Node * donor = mt_population[mt][Rng::randi(mt_population[mt].size())];
        vector<Node*> donor_nodes = donor->subtree();

        for(int & idx : crossover_mask) {
            // check if swap is not necessary
            if (offspring_nodes[idx]->op->sym() == donor_nodes[idx]->op->sym()) {
                // might need to swap if the node is a constant that might be optimized
                if (g::cmut_prob <= 0 || g::cmut_temp <= 0 || donor_nodes[idx]->op->type() != OpType::otConst) {
                    continue;
                }
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
        vector<float> new_fitness = backup_fitness;
        if (change_is_meaningful) {
            // gotta recompute
            new_fitness = g::fit_func->get_fitness_MO(parent);
        }

        // check is not worse
        if (backup_fitness[0]<new_fitness[0]) {
            // undo
            for(int i = 0; i < effectively_changed_indices.size(); i++) {
                int changed_idx = effectively_changed_indices[i];
                Node * off_n = offspring_nodes[changed_idx];
                Op * back_op = backup_ops[i];
                delete off_n->op;
                off_n->op = back_op->clone();
                parent->fitness = backup_fitness;
            }
            parent->trees[mt] = offspring;
            parent->fitness = backup_fitness;
        } else {
            // it improved
            if(new_fitness[0] > backup_fitness[0]) {
                ever_improved = true;
            }

            backup_fitness = new_fitness;
            g::ea->updateMOArchive(parent);
            g::ea->updateSOArchive(parent);
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
                new_fitness = g::fit_func->get_fitness_MO(parent);
            }

            // check is not worse
            if (new_fitness[0] > backup_fitness[0]) {
                // undo
                for(int i = 0; i < effectively_changed_indices.size(); i++) {
                    int changed_idx = effectively_changed_indices[i];
                    Node * off_n = offspring_nodes[changed_idx];
                    Op * back_op = backup_ops[i];
                    delete off_n->op;
                    off_n->op = back_op->clone();
                    offspring->fitness = backup_fitness;
                }
                parent->trees[mt] = offspring;
                parent->fitness = backup_fitness;
            } else{
                if(new_fitness[0] > backup_fitness[0]) {
                    ever_improved = true;
                }

                // it changed and stayed the same or improved
                backup_fitness = new_fitness;
                g::ea->updateMOArchive(parent);
                g::ea->updateSOArchive(parent);
            }

            // discard backup
            for(Op * op : backup_ops) {
                delete op;
            }
        }
    }

    if((macro_generations%g::opt_per_gen)==0 && g::use_optimiser){
        vector<float> fitness_before = parent->fitness;

        Individual * offspring;

        offspring = coeff_opt_lm(parent, true);


        vector<float> fitness_after = g::fit_func->get_fitness_MO(offspring);

        if(fitness_before[0]>fitness_after[0]){
            parent->clear();
            parent = offspring;
            parent->fitness = fitness_after;
            g::ea->updateMOArchive(parent);
            g::ea->updateSOArchive(parent);

            ever_improved = true;
        }
        else{
            offspring->clear();
        }

    }

    // variant of forced improvement that is potentially less aggressive, & less expensive to carry out
    if(g::tournament_size > 1 && !ever_improved){
        // make a tournament between tournament size - 1 candidates + offspring
        vector<Individual*> tournament_candidates;
        tournament_candidates.reserve(g::tournament_size - 1);
        for(int i = 0; i < g::tournament_size - 1; i++) {
            tournament_candidates.push_back(population[Rng::randi(population.size())]);
        }
        tournament_candidates.push_back(parent);

        Individual * winner = tournament(tournament_candidates, g::tournament_size);
        parent->clear();
        parent = winner;
    }

    return parent;
}

// Returns whether offspring dominates previous objective
std::pair<bool, bool>
check_changes_SO(Individual *offspring, vector<float> back_obj, bool FI, int obj) {
    if (offspring->fitness[obj] < back_obj[obj] ) {
        return std::make_pair(true,true);
    }
    if (offspring->fitness[obj] == back_obj[obj] ) {
        if (FI) {
            return std::make_pair(false,false);
        } else {
            return std::make_pair(true,false);
        }
    }

    return std::make_pair(false, false);
}

pair<bool, bool>
check_changes_MO(Individual *offspring, bool FI, vector<float> back_obj){
    bool dominates = false;

    for (size_t j = 0; j < offspring->fitness.size(); j++) {
        if (offspring->fitness[j] < back_obj[j])
            dominates = true;
        else if (offspring->fitness[j] > back_obj[j]) {
            dominates = false;
            break;
        }
    }
    if (dominates) {
        return make_pair(true, true);
    }

    if(!FI) {
        bool same = true;
        for (size_t j = 0; j < offspring->fitness.size(); j++) {
            if (offspring->fitness[j] != back_obj[j]) {
                same = false;
                break;
            }
        }
        if (same) {
            return std::make_pair(true, false);
        }
    }


    // or is not dominated by mo_archive
    if (g::ea->nondominated(offspring)) {
        g::ea->updateMOArchive(offspring);
        return std::make_pair(true,true);
    }

    // dominates, same
    return std::make_pair(false,false);
}

Individual * efficient_gom_MO_FI(Individual * og_parent, vector<pair<vector<int>, int>> & fos, int objective, bool extrema) {
    Individual * parent = og_parent->clone();

    auto random_fos_order = Rng::rand_perm(fos.size());
    vector<float> backup_fitness = parent->fitness;

    bool changed = false;

    for(int fos_idx = 0; fos_idx < fos.size(); fos_idx++){
        int mt = fos[random_fos_order[fos_idx]].second;

        auto crossover_mask = fos[random_fos_order[fos_idx]].first;
        Node * offspring = parent->trees[mt];
        vector<Node*> offspring_nodes = offspring->subtree();

        bool change_is_meaningful = false;
        vector<Op*> backup_ops; backup_ops.reserve(crossover_mask.size());
        vector<int> effectively_changed_indices; effectively_changed_indices.reserve(crossover_mask.size());

        auto donor = extrema ? g::ea->ReturnCopySOMember(objective) : g::ea->ReturnCopyRandomMOMember();
        vector<Node*> donor_nodes = donor->trees[mt]->subtree();

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
        if (change_is_meaningful) {
            //parent->trees[mt] = offspring;

            g::fit_func->get_fitness_MO(parent);

            std::pair<bool, bool> check_changes = extrema ? check_changes_SO(parent, backup_fitness,  true, objective)
                                                          : check_changes_MO(parent, true, backup_fitness);

            if (check_changes.first) {
                // keep changes
                changed = true;
                backup_fitness = parent->fitness;
                g::ea->updateMOArchive(parent);
                g::ea->updateSOArchive(parent);
            }
            else{
                for(int i = 0; i < effectively_changed_indices.size(); i++) {
                    int changed_idx = effectively_changed_indices[i];
                    Node * off_n = offspring_nodes[changed_idx];
                    Op * back_op = backup_ops[i];
                    delete off_n->op;
                    off_n->op = back_op->clone();
                    offspring->fitness = backup_fitness;
                }
                parent->trees[mt] = offspring;
                parent->fitness = backup_fitness;
            }

        }
        else{
            for(int i = 0; i < effectively_changed_indices.size(); i++) {
                int changed_idx = effectively_changed_indices[i];
                Node * off_n = offspring_nodes[changed_idx];
                Op * back_op = backup_ops[i];
                delete off_n->op;
                off_n->op = back_op->clone();
                offspring->fitness = backup_fitness;
            }
            parent->trees[mt] = offspring;
            parent->fitness = backup_fitness;
        }

        // discard backup
        for(Op * op : backup_ops) {
            delete op;
        }

        donor->clear();

        if (changed) {
            break;
        }
    }

    if ((!changed) && !g::ea->MO_archive.empty()) {
        parent->clear();
        if(extrema){
            return g::ea->ReturnCopySOMember(objective);
        }
        else {
            return g::ea->ReturnCopyRandomMOMember();
        }
    }

    return parent;
}

Individual * efficient_gom_MO(Individual * og_parent, vector<vector<Node*>> & population, vector<pair<vector<int>, int>> & fos, int macro_generations, int objective, bool extrema, int NIS_const) {
    Individual * parent = og_parent->clone();

    auto random_fos_order = Rng::rand_perm(fos.size());
    vector<float> backup_fitness = parent->fitness;

    bool changed = false;
    bool ever_improved = false;

    for(int fos_idx = 0; fos_idx < fos.size(); fos_idx++){
        int mt = fos[random_fos_order[fos_idx]].second;

        auto crossover_mask = fos[random_fos_order[fos_idx]].first;
        Node * offspring = parent->trees[mt];
        vector<Node*> offspring_nodes = offspring->subtree();

        bool change_is_meaningful = false;
        vector<Op*> backup_ops; backup_ops.reserve(crossover_mask.size());
        vector<int> effectively_changed_indices; effectively_changed_indices.reserve(crossover_mask.size());

        Node * donor = population[mt][Rng::randi(population[mt].size())];
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
        if (change_is_meaningful) {
            //parent->trees[mt] = offspring;

            g::fit_func->get_fitness_MO(parent);

            // TODO check all below
            std::pair<bool, bool> check_changes = extrema ? check_changes_SO(parent, backup_fitness,  false, objective)
                                                            : check_changes_MO(parent, false, backup_fitness);

//            if(check_changes.second){
//                ever_improved = false;
//            }
            if (check_changes.first) {
                // keep changes
                backup_fitness = parent->fitness;
                changed = true;
                ever_improved = true;
                g::ea->updateMOArchive(parent);
                g::ea->updateSOArchive(parent);
            }
            else{
                for(int i = 0; i < effectively_changed_indices.size(); i++) {
                    int changed_idx = effectively_changed_indices[i];
                    Node * off_n = offspring_nodes[changed_idx];
                    Op * back_op = backup_ops[i];
                    delete off_n->op;
                    off_n->op = back_op->clone();
                    offspring->fitness = backup_fitness;
                }
                parent->trees[mt] = offspring;
                parent->fitness = backup_fitness;
            }

        }
        else{
            for(int i = 0; i < effectively_changed_indices.size(); i++) {
                int changed_idx = effectively_changed_indices[i];
                Node * off_n = offspring_nodes[changed_idx];
                Op * back_op = backup_ops[i];
                delete off_n->op;
                off_n->op = back_op->clone();
                offspring->fitness = backup_fitness;
            }
            parent->trees[mt] = offspring;
            parent->fitness = backup_fitness;
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
            if (change_is_meaningful) {
                // gotta recompute
                g::fit_func->get_fitness_MO(parent);
            }

            // check is not worse
            // Only checking mse because coeffmut cannot alter the number of nodes
            if (parent->fitness[0] > backup_fitness[0]) {
                    // undo
                    for(int i = 0; i < effectively_changed_indices.size(); i++) {
                        int changed_idx = effectively_changed_indices[i];
                        Node *off_n = offspring_nodes[changed_idx];
                        Op *back_op = backup_ops[i];
                        delete off_n->op;
                        off_n->op = back_op->clone();

                    }
                    parent->trees[mt] = offspring;
                    parent->fitness = backup_fitness;
            }else{
                // it improved
                ever_improved = true;
                changed = true;
                backup_fitness = parent->fitness;
                g::ea->updateMOArchive(parent);
                g::ea->updateSOArchive(parent);
            }

            // discard backup ops
            for(Op * op : backup_ops) {
                delete op;
            }
        }
    }

      if((macro_generations%g::opt_per_gen)==0 && g::use_optimiser){
          vector<float> fitness_before = parent->fitness;
          Individual * offspring;

          offspring = coeff_opt_lm(parent, true);

          vector<float> fitness_after = g::fit_func->get_fitness_MO(offspring);

          if(fitness_before[0]>fitness_after[0]){
              parent->clear();
              parent = offspring;
              parent->fitness = fitness_after;
              g::ea->updateMOArchive(parent);
              g::ea->updateSOArchive(parent);
              ever_improved = true;
          }
          else{
              offspring->clear();
          }

    }

    parent->NIS = ever_improved ? 0 : parent->NIS + 1;

    if ((!changed || parent->NIS > NIS_const) && !g::ea->MO_archive.empty()) {
        parent->NIS = 0;
        return efficient_gom_MO_FI(parent, fos, objective, extrema);
    }

    return parent;
}

#endif