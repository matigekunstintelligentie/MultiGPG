#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <iostream>

#include "util.hpp"
#include "myeig.hpp"
#include "globals.hpp"
#include "ims.hpp"

namespace py = pybind11; 
using namespace std;

py::list evolve_val(string options, myeig::Mat &X, myeig::Vec &y, myeig::Mat &X_val, myeig::Vec &y_val) {
    // 1. SETUP
  auto opts = split_string(options, " ");
  int argc = opts.size()+1;
  char * argv[argc];
  string title = "gpg";
  argv[0] = (char*) title.c_str();
  for (int i = 1; i < argc; i++) {
    argv[i] = (char*) opts[i-1].c_str();
  }
  g::read_options(argc, argv);

  // initialize evolution handler 
  IMS * ims = new IMS();

  // set training set
  g::fit_func->set_Xy(X, y);
  g::mse_func->set_Xy(X, y);

  g::fit_func->set_Xy(X_val, y_val, "val");
  g::mse_func->set_Xy(X_val, y_val, "val");
  
  if(g::use_max_range){
    g::set_max_coeff_range();
  }
  // set terminals
  g::set_terminals(g::lib_tset);
  //g::apply_feature_selection(g::lib_feat_sel_number);
  //g::set_terminal_probabilities(g::lib_tset_probs);
  // print("terminal set: ",g::str_terminal_set()," (probs: ",g::lib_tset_probs,")");
  // set batch size
  g::set_batch_size(g::lib_batch_size);
  g::set_batch_size_opt(g::lib_batch_size_opt);
  // print("batch size: ", g::batch_size);
  // print("batch size opt: ", g::batch_size_opt);

  // 2. RUN
  ims->run();

  // 3. OUTPUT
  if (g::ea->MO_archive.empty()) {
    throw runtime_error("Not models found, something went wrong");
  }
  py::list models;
  for (auto model: g::ea->MO_archive) {
    string model_repr = model->human_repr(true);
    models.append(model_repr);
  }

  // 4. CLEANUP
  delete ims;

  return models;
}

py::list evolve(string options, myeig::Mat &X, myeig::Vec &y) {
  // 1. SETUP
  auto opts = split_string(options, " ");
  int argc = opts.size()+1;
  char * argv[argc];
  string title = "gpg";
  argv[0] = (char*) title.c_str();
  for (int i = 1; i < argc; i++) {
    argv[i] = (char*) opts[i-1].c_str();
  }
  g::read_options(argc, argv);

  // initialize evolution handler 
  IMS * ims = new IMS();

  // set training set
  g::fit_func->set_Xy(X, y);
  g::mse_func->set_Xy(X, y);
  
  if(g::use_max_range){
    g::set_max_coeff_range();
  }
  // set terminals
  g::set_terminals(g::lib_tset);
  g::apply_feature_selection(g::lib_feat_sel_number);
  g::set_terminal_probabilities(g::lib_tset_probs);
  print("terminal set: ",g::str_terminal_set()," (probs: ",g::lib_tset_probs,")");
  // set batch size
  g::set_batch_size(g::lib_batch_size);
  g::set_batch_size_opt(g::lib_batch_size_opt);
  // print("batch size: ", g::batch_size);
  // print("batch size opt: ", g::batch_size_opt);

  // 2. RUN
  ims->run();

  // 3. OUTPUT
  if (ims->elites_per_complexity.empty()) {
    throw runtime_error("Not models found, something went wrong");
  }
  py::list models;
  for (auto it = ims->elites_per_complexity.begin(); it != ims->elites_per_complexity.end(); it++) {
    string model_repr = it->second->human_repr();
    models.append(model_repr);
  }

  // 4. CLEANUP
  delete ims;

  return models;
}

PYBIND11_MODULE(_pb_gpg, m) {
  m.doc() = "pybind11-based interface for gpg"; // optional module docstring
  m.def("evolve", &evolve, "Runs gpg evolution in C++");
  m.def("evolve_val", &evolve_val, "Runs gpg evolution in C++");
}
