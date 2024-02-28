#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include <vector>
#include "node.hpp"
#include "util.hpp"

using namespace std;

struct Individual {
  vector<Node*> trees;
  float fitness;

  //Node() {};

  Individual() {
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
      return ind;
  }

  void clear() {
    for(auto tree:trees){
        tree->clear();
    }
    delete this;

  }
  
  int get_num_nodes(bool excl_introns){
      return trees[trees.size()-1]-> get_num_nodes(this->trees, excl_introns);
  }

  Vec get_output(Mat & X, vector<Node *> & trees){
      return trees[trees.size()-1]->get_output(X, trees);
  }

  string human_repr(bool full_tree=false) {
      if(full_tree) {
          return trees[trees.size() - 1]->human_repr(this->trees);
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
