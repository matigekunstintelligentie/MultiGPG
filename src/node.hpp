#ifndef NODE_H
#define NODE_H

#include <vector>
#include "operator.hpp"
#include "util.hpp"
#include "rng.hpp"

#include <sstream>

using namespace std;

struct Node {

  Node * parent = NULL;
  vector<Node*> children;
  Op * op = NULL;
  vector<float> fitness = {9999999,9999999,9999999};

  Node(Op * op) {
    this->op = op;
  }

  virtual ~Node() noexcept(false) {
    if (op)
      delete op;
  }

    void clear() {
        auto nodes = subtree();
        for (int i = 1; i < nodes.size(); i++) {
            delete nodes[i];
        }
        delete this;
    }

  Node * clone() {
    Node * new_node = new Node(this->op->clone());
    new_node->fitness = this->fitness;
    for(Node * c : this->children) {
      Node * new_c = c->clone();
      new_node->append(new_c);
    }
    return new_node;
  }

    int depth() {
        int depth = 0;
        auto * curr = this;
        while(curr->parent) {
            depth++;
            curr = curr->parent;
        }
        return depth;
    }

  void append(Node * c) {
    this->children.push_back(c);
    c->parent = this;
  }

    int get_num_nodes(vector<Node*> &trees, vector<Node*> &fun_children, bool excl_introns=false) {
        int n_nodes = 0;
        if(op->type()==OpType::otFunction){
            n_nodes += trees[((FunctionTree*) op)->id]->get_num_nodes(trees, this->children ,excl_introns);
        }
        else if(op->type()==OpType::otPlaceholder){
            n_nodes += trees[((OutputTree*) op)->id]->get_num_nodes(trees, fun_children ,excl_introns);
        }
        else if(op->type()==OpType::otAny){
            n_nodes += fun_children[((AnyOp*) op)->id]->get_num_nodes(trees, excl_introns);
        }
        else {
            n_nodes = 1;
            int arity = op->arity();
            if(excl_introns) {
                for (int i = 0; i < arity; i++) {
                    n_nodes += children[i]->get_num_nodes(trees, fun_children, excl_introns);
                }
            }
            else{
                for(auto child: children){
                    n_nodes += child->get_num_nodes(trees, fun_children, excl_introns);
                }
            }
        }
        return n_nodes;
    }

  int get_num_nodes(vector<Node*> &trees, bool excl_introns=false) {
      int n_nodes = 0;
      if(op->type()==OpType::otFunction){
          n_nodes += trees[((FunctionTree*) op)->id]->get_num_nodes(trees, this->children, excl_introns);
      }
      else if(op->type()==OpType::otPlaceholder){
          n_nodes += trees[((OutputTree*) op)->id]->get_num_nodes(trees, excl_introns);
      }
      else {
          n_nodes = 1;
          int arity = op->arity();
          if(excl_introns) {
              for (int i = 0; i < arity; i++) {
                  n_nodes += children[i]->get_num_nodes(trees, excl_introns);
              }
          }
          else{
              for(auto child: children){
                  n_nodes += child->get_num_nodes(trees, excl_introns);
              }
          }
      }
      return n_nodes;
  }

  int get_height(vector<Node*> &trees, vector<Node*> &fun_children){
      int height = 0;
      if(op->type()==OpType::otFunction){
          height += trees[((FunctionTree*) op)->id]->get_height(trees, this->children);
      }
      else if(op->type()==OpType::otPlaceholder){
          height += trees[((OutputTree*) op)->id]->get_height(trees, fun_children);
      }
      else if(op->type()==OpType::otAny){
          height += fun_children[((AnyOp*) op)->id]->get_num_nodes(trees);
      }
      else {
          height += 1;
          int arity = op->arity();
          int max_height = 0;
          for (int i = 0; i < arity; i++) {
              int h = children[i]->get_height(trees);
              if(h>max_height){
                  max_height = h;
              }
          }
          height += max_height;
      }
      return height;
  }

  int get_height(vector<Node*> &trees){
      int height = 0;
      if(op->type()==OpType::otFunction){
          height += trees[((FunctionTree*) op)->id]->get_height(trees, this->children);
      }
      else if(op->type()==OpType::otPlaceholder){
          height += trees[((OutputTree*) op)->id]->get_height(trees);
      }
      else {
          height += 1;
          int arity = op->arity();
          int max_height = 0;
          for (int i = 0; i < arity; i++) {
              int h = children[i]->get_height(trees);
              if(h>max_height){
                  max_height = h;
              }
          }
          height += max_height;
      }
      return height;
  }

  // Needed for deleting trees
    vector<Node*> subtree() {
        vector<Node*> subtree;
        _subtree_recursive(subtree);
        return subtree;
    }

    void _subtree_recursive(vector<Node*> &subtree) {
        subtree.push_back(this);
        for(Node * child : children) {
            child->_subtree_recursive(subtree);
        }
    }

  vector<Node*> subtree(vector<Node*> &trees, bool check_introns=false, bool add_ofa=false) {
    vector<Node*> subtree;
    _subtree_recursive(subtree, trees, check_introns, add_ofa);
    return subtree;
  }

    void _subtree_recursive(vector<Node*> &subtree, vector<Node*> &trees,  vector<Node*> &fun_children, bool check_introns, bool add_ofa=false) {
        if(op->type()==OpType::otFunction){
            if(add_ofa){
                subtree.push_back(this);
            }
            trees[((FunctionTree*) op)->id]->_subtree_recursive(subtree, trees, this->children, check_introns, add_ofa);
        }
        else if(op->type()==OpType::otPlaceholder){
            if(add_ofa){
                subtree.push_back(this);
            }
            trees[((OutputTree*) op)->id]->_subtree_recursive(subtree, trees, fun_children, check_introns, add_ofa);
        }
        else if(op->type()==OpType::otAny){
            if(add_ofa){
                subtree.push_back(this);
            }
            fun_children[((AnyOp*) op)->id]->_subtree_recursive(subtree, trees, check_introns, add_ofa);
        }
        else {
            subtree.push_back(this);
            if(!check_introns){
              for (int i=0; i<op->arity(); i++) {
                this->children[i]->_subtree_recursive(subtree, trees, fun_children, check_introns, add_ofa);
              }
            } 
            else{
              for (Node *child: children) {
                child->_subtree_recursive(subtree, trees, fun_children, check_introns, add_ofa);
              }
            }
        }
    }

  void _subtree_recursive(vector<Node*> &subtree, vector<Node*> &trees, bool check_introns, bool add_ofa=false) {
      if(op->type()==OpType::otFunction){
          if(add_ofa){
              subtree.push_back(this);
          }
          trees[((FunctionTree*) op)->id]->_subtree_recursive(subtree, trees, this->children, check_introns, add_ofa);
      }
      else if(op->type()==OpType::otPlaceholder){
          if(add_ofa){
              subtree.push_back(this);
          }
          trees[((OutputTree*) op)->id]->_subtree_recursive(subtree, trees, check_introns, add_ofa);
      }
      else {
          subtree.push_back(this);
          if(!check_introns){
            for (int i=0; i<op->arity(); i++) {
              this->children[i]->_subtree_recursive(subtree, trees, check_introns, add_ofa);
            }
          }
          else{
            for (Node *child: children) {
              child->_subtree_recursive(subtree, trees, check_introns, add_ofa);
            }
          }
      }
  }

    Vec get_output(const Mat & X, vector<Node*> & fun_children, const vector<Node*> & trees) {
        if(op->type()==OpType::otFunction){
            return trees[((FunctionTree*) op)->id]->get_output(X, this->children, trees);
        }
        else if(op->type()==OpType::otPlaceholder){
            return trees[((OutputTree*) op)->id]->get_output(X, fun_children, trees);
        }
        else if(op->type()==OpType::otAny){
            return fun_children[((AnyOp*) op)->id]->get_output(X, trees);
        }
        int a = op->arity();
        if (a == 0) {
            return op->apply(X);
        }

        Mat C(X.rows(), a);
        for(int i = 0; i < a; i++)
            C.col(i) = children[i]->get_output(X, fun_children, trees);

        return op->apply(C);
        //return (op->apply(C) * pow(10.0, NUM_PRECISION)) / (float) pow(10.0,NUM_PRECISION);
    }

  Vec get_output(const Mat & X, const vector<Node*> & trees) {
    if(op->type()==OpType::otFunction){
         return trees[((FunctionTree*) op)->id]->get_output(X, this->children, trees);
    }
    else if(op->type()==OpType::otPlaceholder){
        return trees[((OutputTree*) op)->id]->get_output(X, trees);
    }

    int a = op->arity();
    if (a == 0)
      return op->apply(X);

    Mat C(X.rows(), a);
    for(int i = 0; i < a; i++)
      C.col(i) = children[i]->get_output(X, trees);

    return op->apply(C);
    //return (op->apply(C) * pow(10.0, NUM_PRECISION)) / (float) pow(10.0,NUM_PRECISION);
  }

    float get_complexity_kommenda(vector<float> X, vector<Node*> & fun_children, const vector<Node*> & trees) {
        if(op->type()==OpType::otFunction){
            return trees[((FunctionTree*) op)->id]->get_complexity_kommenda(X, this->children, trees);
        }
        else if(op->type()==OpType::otPlaceholder){
            return trees[((OutputTree*) op)->id]->get_complexity_kommenda(X, fun_children, trees);
        }
        else if(op->type()==OpType::otAny){
            return fun_children[((AnyOp*) op)->id]->get_complexity_kommenda(X, trees);
        }
        int a = op->arity();
        if (a == 0) {
            return op->complexity_kommenda(X);
        }

        vector<float> C;
        for(int i = 0; i < a; i++) {
            C.push_back(children[i]->get_complexity_kommenda(X, fun_children, trees));
        }

        return op->complexity_kommenda(C);
    }

    float get_complexity_kommenda(vector<float> X, const vector<Node*> & trees) {
        if(op->type()==OpType::otFunction){
            return trees[((FunctionTree*) op)->id]->get_complexity_kommenda(X, this->children, trees);
        }
        if(op->type()==OpType::otPlaceholder){
            return trees[((OutputTree*) op)->id]->get_complexity_kommenda(X, trees);
        }
        else if(op->type()==OpType::otAny){
            return 99999999.;
        }

        int a = op->arity();
        if (a == 0)
            return op->complexity_kommenda(X);

        vector<float> C;
        for(int i = 0; i < a; i++) {
            C.push_back(children[i]->get_complexity_kommenda(X, trees));
        }

        return op->complexity_kommenda(C);
    }

    pair<Vec, Vec> get_output_der(const Mat & X, vector<Node*> & fun_children, const vector<Node*> & trees) {
        if(op->type()==OpType::otFunction){
            return trees[((FunctionTree*) op)->id]->get_output_der(X, this->children, trees);
        }
        else if(op->type()==OpType::otPlaceholder){
            return trees[((OutputTree*) op)->id]->get_output_der(X, fun_children, trees);
        }
        else if(op->type()==OpType::otAny){
            return fun_children[((AnyOp*) op)->id]->get_output_der(X, trees);
        }
        int a = op->arity();
        if (a == 0){
            return op->apply_der(X);
        }
        Mat C(X.rows(), a);
        Mat D(X.rows(), a);
        for(int i = 0; i < a; i++){
            pair<Vec, Vec> O = children[i]->get_output_der(X, fun_children, trees);
            C.col(i) = O.first;
            D.col(i) = O.second;
        }
        return op->apply_der(C, D);
    }

  pair<Vec, Vec> get_output_der(const Mat & X, const vector<Node*> & trees) {
    if(op->type()==OpType::otFunction){
        return trees[((FunctionTree*) op)->id]->get_output_der(X, this->children, trees);
    }
    else if(op->type()==OpType::otPlaceholder){
        return trees[((OutputTree*) op)->id]->get_output_der(X, trees);
    }
    int a = op->arity();
    if (a == 0){
        return op->apply_der(X);
    }
    Mat C(X.rows(), a);
    Mat D(X.rows(), a);
    for(int i = 0; i < a; i++){
         pair<Vec, Vec> O = children[i]->get_output_der(X, trees);
         C.col(i) = O.first;
         D.col(i) = O.second;
    }
    return op->apply_der(C, D);
  }


  string str_subtree(vector<Node*> &trees) {
    vector<Node*> nodes = this->subtree(trees);
    string str = "[";
    for(Node * n : nodes) {
      str += n->op->sym() + ", ";
    }
    str.erase(str.end()-2, str.end());
    str += "]";
    return str;
  }

  void print_subtree(vector<Node*> &trees) {
    string str = str_subtree(trees);
    print(str);
  }


    void _human_repr_recursive(string & expr) {

            int arity = op->arity();
            vector<string> args;
            args.reserve(arity);
            for (int i = 0; i < arity; i++) {
                children[i]->_human_repr_recursive(expr);
                args.push_back(expr);
            }

            expr = op->human_repr(args);
    }

    void _human_repr_recursive(vector<Node*> & trees, vector<Node*> fun_children, string & expr, bool add_ofa = true) {
        if(op->type()==OpType::otFunction){

            trees[((FunctionTree*) op)->id]->_human_repr_recursive(trees, this->children, expr, add_ofa);
            if(add_ofa) {
                expr = "func[" + expr + "]";
            }
        }
        else if(op->type()==OpType::otPlaceholder){
            trees[((OutputTree*) op)->id]->_human_repr_recursive(trees,  fun_children, expr, add_ofa);
            if(add_ofa) {
                expr = "out[" + expr + "]";
            }
        }
        else if(op->type()==OpType::otAny){
            fun_children[((AnyOp*) op)->id]->_human_repr_recursive(trees, expr, add_ofa);
            if(add_ofa) {
                expr = "any[" + expr + "]";
            }
        }
        else {
            int arity = op->arity();
            vector<string> args;
            args.reserve(arity);
            for (int i = 0; i < arity; i++) {
                children[i]->_human_repr_recursive(trees,  fun_children,expr, add_ofa);
                args.push_back(expr);
            }
            expr = op->human_repr(args);
        }
    }

  void _human_repr_recursive(vector<Node*> & trees, string & expr, bool add_ofa = true) {
      if(op->type()==OpType::otFunction){

          trees[((FunctionTree*) op)->id]->_human_repr_recursive(trees, this->children, expr, add_ofa);
          if(add_ofa) {
              expr = "func[" + expr + "]";
          }
      }
      else if(op->type()==OpType::otPlaceholder){

          trees[((OutputTree*) op)->id]->_human_repr_recursive(trees, expr, add_ofa);
          if(add_ofa) {
              expr = "out[" + expr + "]";
          }
      }
      else if(op->type()==OpType::otAny){
          //fun_children[((AnyOp*) op)->id]->_human_repr_recursive(trees, expr);
      }
      else {
          int arity = op->arity();
          vector<string> args;
          args.reserve(arity);
          for (int i = 0; i < arity; i++) {
              children[i]->_human_repr_recursive(trees, expr, add_ofa);
              args.push_back(expr);
          }
          expr = op->human_repr(args);
      }
  }

    string human_repr(vector<Node*> & trees, bool add_ofa=true) {
        string result = "";
        _human_repr_recursive(trees, result, add_ofa);
        return result;
    }

  string human_repr() {
    string result = "";
    _human_repr_recursive(result);
    return result;
  }

  void _np_repr_recursive(string & expr) {
    int arity = op->arity();
    vector<string> args; args.reserve(arity);
    for(int i = 0; i < arity; i++) {
      children[i]->_np_repr_recursive(expr);
      args.push_back(expr);
    }
    expr = op->np_repr(args);
  }

  string np_repr() {
    string result = "";
    _np_repr_recursive(result);
    return result;
  }

  void _torch_repr_recursive(string & expr) {
    int arity = op->arity();
    vector<string> args; args.reserve(arity);
    for(int i = 0; i < arity; i++) {
      children[i]->_torch_repr_recursive(expr);
      args.push_back(expr);
    }
    expr = op->torch_repr(args);
  }

  string torch_repr() {
    string result = "";
    _torch_repr_recursive(result);
    return result;
  }

};




#endif
