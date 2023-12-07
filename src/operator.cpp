#include "operator.hpp"
#include "node.hpp"

 
string OutputTree::human_repr(vector<string> & args)  {
  print(to_string(id) + " output");
  return this->tree->human_repr();
}

Op * OutputTree::clone(){
  print("sldjksj");
  return new OutputTree(this->id, this->tree);
}
 
int OutputTree::arity()  {
  return this->tree->op->arity();
}
 
string OutputTree::sym()  {
  print("symba");
  return "Output(" + this->tree->op->sym() + ")";
}
 
Vec OutputTree::apply(Mat & X)  {
  return this->tree->get_output(X);
}
 
pair<Vec, Vec> OutputTree::apply_der(Mat & X)  {
  return this->tree->get_output_der(X);
}

string OutputTree::np_repr(vector<string> & args){
  return this->tree->np_repr();
}
 
string OutputTree::torch_repr(vector<string> & args){
    return this->tree->torch_repr();
}