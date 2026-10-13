#ifndef NODE_H
#define NODE_H

#include <vector>
#include <unordered_map>
#include "operator.hpp"
#include "util.hpp"
#include "rng.hpp"

#include <sstream>

using namespace std;

struct Node;

struct Tape {
  Node * node = NULL;
  Mat C;
  vector<Tape> kids;
};

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

    int get_num_nodes(vector<Node*> &trees, const vector<int> &arg_counts, bool excl_introns=false) {
        int n_nodes = 0;
        if(op->type()==OpType::otFunction){
            vector<int> new_arg_counts(children.size());
            for(size_t i = 0; i < children.size(); i++){
                new_arg_counts[i] = children[i]->get_num_nodes(trees, arg_counts, excl_introns);
            }
            n_nodes += trees[((FunctionTree*) op)->id]->get_num_nodes(trees, new_arg_counts, excl_introns);
        }
        else if(op->type()==OpType::otPlaceholder){
            n_nodes += trees[((OutputTree*) op)->id]->get_num_nodes(trees, arg_counts ,excl_introns);
        }
        else if(op->type()==OpType::otAny){
            n_nodes += arg_counts[((AnyOp*) op)->id];
        }
        else {
            n_nodes = 1;
            int arity = op->arity();
            if(excl_introns) {
                for (int i = 0; i < arity; i++) {
                    n_nodes += children[i]->get_num_nodes(trees, arg_counts, excl_introns);
                }
            }
            else{
                for(auto child: children){
                    n_nodes += child->get_num_nodes(trees, arg_counts, excl_introns);
                }
            }
        }
        return n_nodes;
    }

  int get_num_nodes(vector<Node*> &trees, bool excl_introns=false) {
      int n_nodes = 0;
      if(op->type()==OpType::otFunction){
          vector<int> arg_counts(children.size());
          for(size_t i = 0; i < children.size(); i++){
              arg_counts[i] = children[i]->get_num_nodes(trees, excl_introns);
          }
          n_nodes += trees[((FunctionTree*) op)->id]->get_num_nodes(trees, arg_counts, excl_introns);
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

  int num_nodes_capped(vector<Node*> &trees, const vector<int> &arg_counts, int cap) {
      if(op->type()==OpType::otFunction){
          vector<int> new_arg_counts(children.size());
          for(size_t i = 0; i < children.size(); i++)
              new_arg_counts[i] = children[i]->num_nodes_capped(trees, arg_counts, cap);
          return trees[((FunctionTree*) op)->id]->num_nodes_capped(trees, new_arg_counts, cap);
      }
      if(op->type()==OpType::otPlaceholder)
          return trees[((OutputTree*) op)->id]->num_nodes_capped(trees, arg_counts, cap);
      if(op->type()==OpType::otAny)
          return arg_counts[((AnyOp*) op)->id];
      int a = op->arity();
      if(a == 0) return 1;
      int total = 1;
      for(int i = 0; i < a; i++){
          total += children[i]->num_nodes_capped(trees, arg_counts, cap);
          if(total > cap) return cap + 1;
      }
      return total;
  }

  int num_nodes_capped(vector<Node*> &trees, int cap) {
      if(op->type()==OpType::otFunction){
          vector<int> arg_counts(children.size());
          for(size_t i = 0; i < children.size(); i++)
              arg_counts[i] = children[i]->num_nodes_capped(trees, cap);
          return trees[((FunctionTree*) op)->id]->num_nodes_capped(trees, arg_counts, cap);
      }
      if(op->type()==OpType::otPlaceholder)
          return trees[((OutputTree*) op)->id]->num_nodes_capped(trees, cap);
      int a = op->arity();
      if(a == 0) return 1;
      int total = 1;
      for(int i = 0; i < a; i++){
          total += children[i]->num_nodes_capped(trees, cap);
          if(total > cap) return cap + 1;
      }
      return total;
  }

  int get_height(vector<Node*> &trees, const vector<int> &arg_heights){
      int height = 0;
      if(op->type()==OpType::otFunction){
          vector<int> new_arg_heights(children.size());
          for(size_t i = 0; i < children.size(); i++){
              new_arg_heights[i] = children[i]->get_height(trees, arg_heights);
          }
          height += trees[((FunctionTree*) op)->id]->get_height(trees, new_arg_heights);
      }
      else if(op->type()==OpType::otPlaceholder){
          height += trees[((OutputTree*) op)->id]->get_height(trees, arg_heights);
      }
      else if(op->type()==OpType::otAny){
          height += arg_heights[((AnyOp*) op)->id];
      }
      else {
          height += 1;
          int arity = op->arity();
          int max_height = 0;
          for (int i = 0; i < arity; i++) {
              int h = children[i]->get_height(trees, arg_heights);
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
          vector<int> arg_heights(children.size());
          for(size_t i = 0; i < children.size(); i++){
              arg_heights[i] = children[i]->get_height(trees);
          }
          height += trees[((FunctionTree*) op)->id]->get_height(trees, arg_heights);
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

    void _subtree_recursive(vector<Node*> &subtree, vector<Node*> &trees,  const vector<vector<Node*>> &arg_nodes, bool check_introns, bool add_ofa=false) {
        if(op->type()==OpType::otFunction){
            if(add_ofa){
                subtree.push_back(this);
            }
            vector<vector<Node*>> new_arg_nodes(children.size());
            for(size_t i = 0; i < children.size(); i++){
                children[i]->_subtree_recursive(new_arg_nodes[i], trees, arg_nodes, check_introns, add_ofa);
            }
            trees[((FunctionTree*) op)->id]->_subtree_recursive(subtree, trees, new_arg_nodes, check_introns, add_ofa);
        }
        else if(op->type()==OpType::otPlaceholder){
            if(add_ofa){
                subtree.push_back(this);
            }
            trees[((OutputTree*) op)->id]->_subtree_recursive(subtree, trees, arg_nodes, check_introns, add_ofa);
        }
        else if(op->type()==OpType::otAny){
            if(add_ofa){
                subtree.push_back(this);
            }
            const vector<Node*> &arg = arg_nodes[((AnyOp*) op)->id];
            subtree.insert(subtree.end(), arg.begin(), arg.end());
        }
        else {
            subtree.push_back(this);
            if(!check_introns){
              for (int i=0; i<op->arity(); i++) {
                this->children[i]->_subtree_recursive(subtree, trees, arg_nodes, check_introns, add_ofa);
              }
            }
            else{
              for (Node *child: children) {
                child->_subtree_recursive(subtree, trees, arg_nodes, check_introns, add_ofa);
              }
            }
        }
    }

  void _subtree_recursive(vector<Node*> &subtree, vector<Node*> &trees, bool check_introns, bool add_ofa=false) {
      if(op->type()==OpType::otFunction){
          if(add_ofa){
              subtree.push_back(this);
          }
          vector<vector<Node*>> arg_nodes(children.size());
          for(size_t i = 0; i < children.size(); i++){
              children[i]->_subtree_recursive(arg_nodes[i], trees, check_introns, add_ofa);
          }
          trees[((FunctionTree*) op)->id]->_subtree_recursive(subtree, trees, arg_nodes, check_introns, add_ofa);
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

    Vec get_output(const Mat & X, const Mat & argvals, const vector<Node*> & trees) {
        if(op->type()==OpType::otFunction){
            Mat new_argvals(X.rows(), children.size());
            for(size_t i = 0; i < children.size(); i++){
                new_argvals.col(i) = children[i]->get_output(X, argvals, trees);
            }
            return trees[((FunctionTree*) op)->id]->get_output(X, new_argvals, trees);
        }
        else if(op->type()==OpType::otPlaceholder){
            return trees[((OutputTree*) op)->id]->get_output(X, argvals, trees);
        }
        else if(op->type()==OpType::otAny){
            return argvals.col(((AnyOp*) op)->id);
        }
        int a = op->arity();
        if (a == 0) {
            return op->apply(X);
        }

        Mat C(X.rows(), a);
        for(int i = 0; i < a; i++)
            C.col(i) = children[i]->get_output(X, argvals, trees);

        return op->apply(C);
        //return (op->apply(C) * pow(10.0, NUM_PRECISION)) / (float) pow(10.0,NUM_PRECISION);
    }

  Vec get_output(const Mat & X, const vector<Node*> & trees) {
    if(op->type()==OpType::otFunction){
        Mat argvals(X.rows(), children.size());
        for(size_t i = 0; i < children.size(); i++){
            argvals.col(i) = children[i]->get_output(X, trees);
        }
        return trees[((FunctionTree*) op)->id]->get_output(X, argvals, trees);
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

    float get_complexity_kommenda(vector<float> X, const vector<float> & arg_complexities, const vector<Node*> & trees) {
        if(op->type()==OpType::otFunction){
            vector<float> new_arg_complexities(children.size());
            for(size_t i = 0; i < children.size(); i++){
                new_arg_complexities[i] = children[i]->get_complexity_kommenda(X, arg_complexities, trees);
            }
            return trees[((FunctionTree*) op)->id]->get_complexity_kommenda(X, new_arg_complexities, trees);
        }
        else if(op->type()==OpType::otPlaceholder){
            return trees[((OutputTree*) op)->id]->get_complexity_kommenda(X, arg_complexities, trees);
        }
        else if(op->type()==OpType::otAny){
            return arg_complexities[((AnyOp*) op)->id];
        }
        int a = op->arity();
        if (a == 0) {
            return op->complexity_kommenda(X);
        }

        vector<float> C;
        for(int i = 0; i < a; i++) {
            C.push_back(children[i]->get_complexity_kommenda(X, arg_complexities, trees));
        }

        return op->complexity_kommenda(C);
    }

    float get_complexity_kommenda(vector<float> X, const vector<Node*> & trees) {
        if(op->type()==OpType::otFunction){
            vector<float> arg_complexities(children.size());
            for(size_t i = 0; i < children.size(); i++){
                arg_complexities[i] = children[i]->get_complexity_kommenda(X, trees);
            }
            return trees[((FunctionTree*) op)->id]->get_complexity_kommenda(X, arg_complexities, trees);
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

    pair<Vec, Vec> get_output_der(const Mat & X, const Mat & argvals, const Mat & argders, const vector<Node*> & trees) {
        if(op->type()==OpType::otFunction){
            Mat new_vals(X.rows(), children.size());
            Mat new_ders(X.rows(), children.size());
            for(size_t i = 0; i < children.size(); i++){
                pair<Vec, Vec> O = children[i]->get_output_der(X, argvals, argders, trees);
                new_vals.col(i) = O.first;
                new_ders.col(i) = O.second;
            }
            return trees[((FunctionTree*) op)->id]->get_output_der(X, new_vals, new_ders, trees);
        }
        else if(op->type()==OpType::otPlaceholder){
            return trees[((OutputTree*) op)->id]->get_output_der(X, argvals, argders, trees);
        }
        else if(op->type()==OpType::otAny){
            int k = ((AnyOp*) op)->id;
            return make_pair(argvals.col(k), argders.col(k));
        }
        int a = op->arity();
        if (a == 0){
            return op->apply_der(X);
        }
        Mat C(X.rows(), a);
        Mat D(X.rows(), a);
        for(int i = 0; i < a; i++){
            pair<Vec, Vec> O = children[i]->get_output_der(X, argvals, argders, trees);
            C.col(i) = O.first;
            D.col(i) = O.second;
        }
        if ((D == 0).all())
            return make_pair(op->apply(C), Vec::Zero(X.rows()));
        return op->apply_der(C, D);
    }

  pair<Vec, Vec> get_output_der(const Mat & X, const vector<Node*> & trees) {
    if(op->type()==OpType::otFunction){
        Mat argvals(X.rows(), children.size());
        Mat argders(X.rows(), children.size());
        for(size_t i = 0; i < children.size(); i++){
            pair<Vec, Vec> O = children[i]->get_output_der(X, trees);
            argvals.col(i) = O.first;
            argders.col(i) = O.second;
        }
        return trees[((FunctionTree*) op)->id]->get_output_der(X, argvals, argders, trees);
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
    if ((D == 0).all())
      return make_pair(op->apply(C), Vec::Zero(X.rows()));
    return op->apply_der(C, D);
  }

  Vec get_output_tape(const Mat & X, const Mat & argvals, const vector<Node*> & trees, Tape & t) {
    t.node = this;
    if(op->type()==OpType::otFunction){
        Mat new_vals(X.rows(), children.size());
        t.kids.resize(children.size() + 1);
        for(size_t i = 0; i < children.size(); i++){
            new_vals.col(i) = children[i]->get_output_tape(X, argvals, trees, t.kids[i]);
        }
        return trees[((FunctionTree*) op)->id]->get_output_tape(X, new_vals, trees, t.kids.back());
    }
    else if(op->type()==OpType::otPlaceholder){
        t.kids.resize(1);
        return trees[((OutputTree*) op)->id]->get_output_tape(X, argvals, trees, t.kids[0]);
    }
    else if(op->type()==OpType::otAny){
        return argvals.col(((AnyOp*) op)->id);
    }
    int a = op->arity();
    if (a == 0)
        return op->apply(X);
    Mat C(X.rows(), a);
    t.kids.resize(a);
    for(int i = 0; i < a; i++){
        C.col(i) = children[i]->get_output_tape(X, argvals, trees, t.kids[i]);
    }
    Vec out = op->apply(C);
    t.C = std::move(C);
    return out;
  }

  void backprop(const Vec & G, Tape & t, Mat & argadj, unordered_map<Node*, int> & coeff_col, Mat & J) {
    if ((G == 0).all())
        return;
    if(op->type()==OpType::otFunction){
        Mat new_adj = Mat::Zero(G.rows(), children.size());
        t.kids.back().node->backprop(G, t.kids.back(), new_adj, coeff_col, J);
        for(size_t i = 0; i < children.size(); i++){
            children[i]->backprop(new_adj.col(i), t.kids[i], argadj, coeff_col, J);
        }
        return;
    }
    else if(op->type()==OpType::otPlaceholder){
        t.kids[0].node->backprop(G, t.kids[0], argadj, coeff_col, J);
        return;
    }
    else if(op->type()==OpType::otAny){
        argadj.col(((AnyOp*) op)->id) += G;
        return;
    }
    int a = op->arity();
    if (a == 0) {
        if (op->type() == OpType::otConst) {
            auto it = coeff_col.find(this);
            if (it != coeff_col.end())
                J.col(it->second) += G;
        }
        return;
    }
    Mat A = op->apply_back(t.C, G);
    for(int i = 0; i < a; i++){
        children[i]->backprop(A.col(i), t.kids[i], argadj, coeff_col, J);
    }
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

    void _human_repr_recursive(vector<Node*> & trees, const vector<string> & arg_reprs, string & expr, bool add_ofa = true) {
        if(op->type()==OpType::otFunction){
            vector<string> new_arg_reprs(children.size());
            for(size_t i = 0; i < children.size(); i++){
                string arg_expr = "";
                children[i]->_human_repr_recursive(trees, arg_reprs, arg_expr, add_ofa);
                new_arg_reprs[i] = arg_expr;
            }
            trees[((FunctionTree*) op)->id]->_human_repr_recursive(trees, new_arg_reprs, expr, add_ofa);
            if(add_ofa) {
                expr = "func[" + expr + "]";
            }
        }
        else if(op->type()==OpType::otPlaceholder){
            trees[((OutputTree*) op)->id]->_human_repr_recursive(trees,  arg_reprs, expr, add_ofa);
            if(add_ofa) {
                expr = "out[" + expr + "]";
            }
        }
        else if(op->type()==OpType::otAny){
            expr = arg_reprs[((AnyOp*) op)->id];
            if(add_ofa) {
                expr = "any[" + expr + "]";
            }
        }
        else {
            int arity = op->arity();
            vector<string> args;
            args.reserve(arity);
            for (int i = 0; i < arity; i++) {
                children[i]->_human_repr_recursive(trees,  arg_reprs,expr, add_ofa);
                args.push_back(expr);
            }
            expr = op->human_repr(args);
        }
    }

  void _human_repr_recursive(vector<Node*> & trees, string & expr, bool add_ofa = true) {
      if(op->type()==OpType::otFunction){
          vector<string> arg_reprs(children.size());
          for(size_t i = 0; i < children.size(); i++){
              string arg_expr = "";
              children[i]->_human_repr_recursive(trees, arg_expr, add_ofa);
              arg_reprs[i] = arg_expr;
          }
          trees[((FunctionTree*) op)->id]->_human_repr_recursive(trees, arg_reprs, expr, add_ofa);
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
