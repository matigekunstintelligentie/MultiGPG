#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include <vector>
#include "node.hpp"
#include "util.hpp"
#include <set>

using namespace std;

struct Individual {
  vector<Node*> trees;
  vector<float> fitness = {9999999,9999999};
  int clusterid;
  int NIS = 0;
  float add = 0.;
  float mul = 1.;

  
  Individual(){
  }

  virtual ~Individual() noexcept(false) {
  }
  
  Individual * clone(){
      Individual * ind = new Individual();
      ind->trees.reserve(trees.size());
      for(auto tree:trees){
          ind->trees.push_back(tree->clone());
      }
      ind->fitness = this->fitness;
      ind->NIS = this->NIS;
      ind->add = this->add;
      ind->mul = this->mul;
      return ind;
  }

  void clear() {
    for(auto tree:trees){
        tree->clear();
    }
    delete this;

  }

  vector<Node*> subtree(bool check_introns){
      return  trees[trees.size()-1]->subtree(this->trees, check_introns);
  }

  bool is_intron(Node * node){
      for(Node * n: this->subtree(false)){
          if(n == node){
              return false;
          }
      }
      return true;
  }
  
  int get_num_nodes(bool excl_introns, bool discount=false){

      if(!discount){
          return trees[trees.size()-1]-> get_num_nodes(this->trees, excl_introns);
      }
      else{
          vector<Node*> nodes = trees[trees.size()-1]->subtree(this->trees, !excl_introns);

          vector<Node*> node_vec;
          int count = 0;
          for(auto node: nodes){
              if(node->op->type()==OpType::otConst || node->op->type()==OpType::otFeat){
                  count++;
              }
              else{
                  node_vec.push_back(node);
              }
          }
          set<Node*> node_set;
          for(auto node: node_vec) {
              node_set.insert(node);
          }
          return static_cast<int>(node_set.size()) + count;
      }
  }

  Vec get_output(const Mat & X){
      return trees[trees.size()-1]->get_output(X, this->trees);
  }

  pair<Vec, Vec> get_output_der(const Mat & X){
    return trees[trees.size()-1]->get_output_der(X, this->trees);
  }

  string human_repr(bool full_tree=false, bool add_ofa = true) {
      if(full_tree) {
          return trees[trees.size() - 1]->human_repr(this->trees, add_ofa);
      }
      else{
          return trees[trees.size() - 1]->human_repr();
      }
  }

  string np_repr() {
    return trees[trees.size()-1]->np_repr();
  }

};


#endif
