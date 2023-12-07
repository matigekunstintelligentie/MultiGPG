#include "node.hpp"

  Node::Node(Op * op) {
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

  Node * Node::clone() {
    Node * new_node = new Node(this->op->clone());
    new_node->fitness = this->fitness;
    new_node->improvement_opt = this->improvement_opt;
    new_node->improvement_coeff = this->improvement_coeff;
    new_node->improvement_gom = this->improvement_gom;
    for(Node * c : this->children) {
      Node * new_c = c->clone();
      new_node->append(new_c);
    }
    return new_node;
  }

  void Node::append(Node * c) {
    this->children.push_back(c);
    c->parent = this;
  }

  vector<Node*>::iterator Node::detach(Node * c){
    auto it = children.begin();
    for(auto it = children.begin(); it < children.end(); it++){
      if ((*it)==c){
        break;
      }
    }
    assert(it != children.end());
    children.erase(it);
    c->parent = NULL;
    return it;
  }

  Node * Node::detach(int idx) {
    assert(idx < children.size());
    auto it = children.begin() + idx;
    Node * c = children[idx];
    children.erase(it);
    c->parent = NULL;
    return c;
  }

  void Node::insert(Node * c, vector<Node*>::iterator it) {
    children.insert(it, c);
    c->parent = this;
  }

  int Node::depth() {
    int depth = 0;
    auto * curr = this;
    while(curr->parent) {
      depth++;
      curr = curr->parent;
    }
    return depth;
  }

  int Node::height() {
    int max_child_depth = 0;
    _height_recursive(max_child_depth);
    int h = max_child_depth - depth();
    assert(h >= 0);
    return h;
  }

  int Node::get_num_nodes(bool excl_introns=false) {
    auto nodes = this->subtree();
    int n = nodes.size();
    if (!excl_introns) 
      return n;

    int num_introns = 0;
    for(Node * n : nodes) {
      if (n->is_intron()) {
        num_introns++;
      }
    }
    return n - num_introns;
  }

  void Node::_height_recursive(int & max_child_depth) {
    if (op->arity() == 0) {
      int d = this->depth();
      if (d > max_child_depth)
        max_child_depth = d;
    }

    for (int i = 0; i < op->arity(); i++)
      children[i]->_height_recursive(max_child_depth);
  }

  vector<Node*> Node::subtree() {
    vector<Node*> subtree;
    subtree.reserve(64);
    _subtree_recursive(subtree);
    return subtree;
  }

  void Node::_subtree_recursive(vector<Node*> &subtree) {
    subtree.push_back(this);
    for(Node * child : children) {
      child->_subtree_recursive(subtree);
    }
  }

  vector<Node*> Node::subtree(bool check_introns) {
    vector<Node*> subtree;
    subtree.reserve(64);
    _subtree_recursive(subtree, check_introns);
    return subtree;
  }

  void Node::_subtree_recursive(vector<Node*> &subtree, bool check_introns) {
    if(check_introns){
      if(!this->is_intron()){
        subtree.push_back(this);
        for(Node * child : children) {
          child->_subtree_recursive(subtree, check_introns);
        }
      }
    }
    else{
      subtree.push_back(this);
      for(Node * child : children) {
        child->_subtree_recursive(subtree, check_introns);
      }
    }

  }

  int Node::position_among_siblings() {
    if (!parent)
      return 0;
    int i = 0;
    for (Node * s : parent->children) {
      if (s == this)
        return i;
      i++;
    }
    throw runtime_error("Unreachable code");
  }

  bool Node::is_intron() {
    Node * p = parent;
    if(!p)
      return false;
    Node * n = this;
    while (p) {
      if (n->position_among_siblings() >= p->op->arity())
        return true;
      n = p;
      p = n->parent;
    }
    return false;
  }

  Vec Node::get_output(Mat & X) {
    int a = op->arity();
    if (a == 0)
      return op->apply(X);

    Mat C(X.rows(), a);
    for(int i = 0; i < a; i++)
      C.col(i) = children[i]->get_output(X);

    return op->apply(C);
    //return (op->apply(C) * pow(10.0, NUM_PRECISION)) / (float) pow(10.0,NUM_PRECISION);
  }


  pair<Vec, Vec> Node::get_output_der(Mat & X) {
    int a = op->arity();
    if (a == 0){
        return op->apply_der(X);
    }
    Mat C(X.rows(), a);
    Mat D(X.rows(), a);
    for(int i = 0; i < a; i++){
         pair<Vec, Vec> O = children[i]->get_output_der(X);
         C.col(i) = O.first;
         D.col(i) = O.second;
    }
    return op->apply_der(C, D);
  }


  string Node::str_subtree() {
    vector<Node*> nodes = this->subtree();
    string str = "[";
    for(Node * n : nodes) {
      str += n->op->sym() + ", ";
    }
    str.erase(str.end()-2, str.end());
    str += "]";
    return str;
  }

  void Node::print_subtree() {
    string str = str_subtree();
    print(str);
  }


  void Node::_human_repr_recursive(string & expr) {
    int arity = op->arity();
    vector<string> args; args.reserve(arity);
    for(int i = 0; i < arity; i++) {
      children[i]->_human_repr_recursive(expr);
      args.push_back(expr);
    }
    expr = op->human_repr(args);
  }

  string human_repr() {
    string result = "";
    _human_repr_recursive(result);
    return result;
  }

  void Node::_np_repr_recursive(string & expr) {
    int arity = op->arity();
    vector<string> args; args.reserve(arity);
    for(int i = 0; i < arity; i++) {
      children[i]->_np_repr_recursive(expr);
      args.push_back(expr);
    }
    expr = op->np_repr(args);
  }

  string Node::np_repr() {
    string result = "";
    _np_repr_recursive(result);
    return result;
  }

  void Node::_torch_repr_recursive(string & expr) {
    int arity = op->arity();
    vector<string> args; args.reserve(arity);
    for(int i = 0; i < arity; i++) {
      children[i]->_torch_repr_recursive(expr);
      args.push_back(expr);
    }
    expr = op->torch_repr(args);
  }

  string Node::torch_repr() {
    string result = "";
    _torch_repr_recursive(result);
    return result;
  }