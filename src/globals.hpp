#ifndef GLOBALS_H
#define GLOBALS_H

#include <random>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "myeig.hpp"
#include "operator.hpp"
#include "fitness.hpp"
#include "cmdparser.hpp"
#include "feature_selection.hpp"
#include "rng.hpp"
#include <dlib/rand.h>

using namespace std;
using namespace myeig;

namespace g {

  // ALL operators
  vector<Op*> all_operators = {
    new Add(), new Sub(), new Neg(), new Mul(), new Div(), new Inv(), 
    new Square(), new Sqrt(), new Cube(),
    new Sin(), new Cos(), 
    new Log(), new Pow(), new Max(), new Min(), new Exp(), new Abs(),
  };

  // ALL fitness functions 
  vector<Fitness*> all_fitness_functions = {
    new MAEFitness(), new MSEFitness(), new AbsCorrFitness(), new LSMSEFitness()
  };

  // budget
  int pop_size;
  int max_generations;
  int max_time;
  int max_evaluations;
  long long max_node_evaluations;
  bool disable_ims = false;

  int jacobian_evals = 0;

  // joe
  int nr_multi_trees;
  bool use_optimiser=false;
  bool use_local_search=false;
  string optimiser_choice;
  int opt_per_gen;
  string csv_file;
  bool use_clip=false;
  bool reinject_elite=false;

  float tol;
  bool use_ftol=false;
  bool use_mse_opt=false;
  int lm_max_fev = 10;
  int bfgs_max_iter = 2;
  int nr_improvements = 0;
  float amount_improvements = 0.;
  bool log=false;
  bool optimise_after = false;
  bool add_addition_multiplication = false;
  bool add_any = false;
  float range = 10.;
  bool use_max_range=false;
  bool equal_p_coeffs=false;
  float random_accept_p = 0.0;


  // representation
  int max_depth;
  string init_strategy;
  vector<Op*> functions;
  vector<Op*> terminals;
  Vec cumul_fset_probs;
  Vec cumul_tset_probs;
  string lib_tset; // used when `fit` is called when using as lib
  string lib_tset_probs; // used when `fit` is called when using as lib
  string complexity_type;
  float rel_compl_importance=0.0;
  int lib_feat_sel_number = -1; // used when `fit` is called when using as lib

  // problem
  Fitness * fit_func = NULL;
  Fitness * mse_func = NULL;

  string path_to_training_set;
  string lib_batch_size; // used when `fit` is called when using as lib
  int batch_size;
  string lib_batch_size_opt;
  int batch_size_opt;

  // variation
  int max_init_attempts = 10000;
  bool no_linkage;
  float cmut_eps;
  float cmut_prob;
  float cmut_temp;
  bool no_large_subsets=false;
  bool no_univariate=false;
  bool no_univariate_except_leaves=false;



  // selection
  int tournament_size;
  bool tournament_stochastic = false;

  // other
  int random_state = -1;
  bool verbose = true;
  bool _call_as_lib = false;

  // Functions
  void set_fit_func(string fit_func_name) { 
    bool found = false;
    for (auto * f : all_fitness_functions) {
      if (f->name() == fit_func_name) {
        found = true;
        fit_func = f->clone();
      }
      if(f->name()=="mse"){
          mse_func = f->clone();
      }
    }
    if (!found) {
      throw runtime_error("Unrecognized fitness function: "+fit_func_name);
    }
  }

  void set_functions(string setting) {
    assert(functions.empty());
    vector<string> desired_operator_symbs = split_string(setting);
    for (string sym : desired_operator_symbs) {
      bool found = false;
      for (Op * op : all_operators) {
        if (op->sym() == sym) {
          found = true;
          functions.push_back(op->clone());
          break;
        }
      }
      if (!found) {
        throw runtime_error("Unrecognized function: "+sym);
      }
    }

  }

  Vec _compute_custom_cumul_probs_operator_set(string setting, vector<Op*> & op_set) {

    auto str_v = split_string(setting);
    if (str_v.size() != op_set.size()) {
      throw runtime_error("Size of the probabilities for function or terminal set does not match the size of the respective set: "+setting);
    }

    Vec result(op_set.size());
    float cumul_prob = 0;
    for (int i = 0; i < str_v.size(); i++) {
      cumul_prob += stof(str_v[i]);
      result[i] = cumul_prob;
    }

    if (abs(1.0-result[result.size()-1]) > 1e-3) {
      throw runtime_error("The probabilties for the respective operator set do not sum to 1: "+setting);
    }

    return result;
  }

  void set_function_probabilities(string setting) {
      
    if (setting == "auto") {
      // set unary operators to have half the chance other ones (which are normally binary)
      Veci arities(functions.size());
      int num_unary = 0;
      for(int i = 0; i < functions.size(); i++) {
        arities[i] = functions[i]->arity();
        if (arities[i] == 1) {
          num_unary++;
        }
      }

      int num_other = functions.size() - num_unary;
      float p_unary = 1.0 / (2.0*num_other + num_unary);
      float p_other = 1.0 / (num_other + 0.5*num_unary);

      float cumul_prob = 0;
      cumul_fset_probs = Vec(functions.size());
      for (int i = 0; i < arities.size(); i++) {
        if (arities[i] == 1) {
          cumul_prob += p_unary;
        } else {
          cumul_prob += p_other;
        }
        cumul_fset_probs[i] = cumul_prob;
      }
      return;
    }

    // else, use what provided
    cumul_fset_probs = _compute_custom_cumul_probs_operator_set(setting, functions);
  }

  void set_max_coeff_range(){
    float maxc = abs(fit_func->X_train.maxCoeff());
    float minc = abs(fit_func->X_train.minCoeff());
    range = max(maxc,minc);
  }

  void set_terminals(string setting) {
    assert(terminals.empty());

    if (setting == "auto") {
      assert(fit_func);
      for(int i = 0; i < fit_func->X_train.cols(); i++) {
        terminals.push_back(new Feat(i));
      }
      if(equal_p_coeffs){
        for(int i = 0; i < fit_func->X_train.cols(); i++) {
          terminals.push_back(new Const());
        }
      }
      else{
        terminals.push_back(new Const());
      }
    }
    else {
      vector<string> desired_terminal_symbs = split_string(setting);
      for (string sym : desired_terminal_symbs) {
        try {
          if (sym.size() > 2 && sym.substr(0, 2) == "x_") {
            // variable
            int i = stoi(sym.substr(2,sym.size()));
            terminals.push_back(new Feat(i)); 
          } else if (sym == "erc") {
            terminals.push_back(new Const(NAN, range));
          } else {
            // constant
            float c = stof(sym);
            terminals.push_back(new Const(c));
          }
        } catch(std::invalid_argument const& ex) {
          throw runtime_error("Unrecognized terminal: "+sym);
        }
      }
    }
  }



  void set_terminal_probabilities(string setting) {
    print("set_terminal_probabilities");
    if (setting == "auto") {
      cumul_tset_probs = Vec(terminals.size());
      float p = 1.0 / terminals.size();
      float cumul_p = 0;
      for (int i = 0; i < terminals.size(); i++) {
        cumul_p += p;
        cumul_tset_probs[i] = cumul_p;
      }
      return;
    }
    // else, use what provided
    cumul_tset_probs = _compute_custom_cumul_probs_operator_set(setting, terminals);
  }

  string str_terminal_set() {
    string str = "";
    for (Op * el : terminals) {
      if (el->type() == OpType::otConst && isnan(((Const*)el)->c)) {
        str += "erc,";
      } else {
        str += el->sym() + ",";
      }
    }
    str = str.substr(0, str.size()-1);
    return str;
  }

  void set_batch_size(string lib_batch_size) {
    if (lib_batch_size == "auto") {
      batch_size = fit_func->X_train.rows();
    } else {
      int n = fit_func->X_train.rows();
      batch_size = stoi(lib_batch_size);
      if (batch_size > n) {
        //print("[!] Warning: batch size is larger than the number of training examples. Setting it to ", n);
        batch_size = n;
      }
    }
  }

  void set_batch_size_opt(string lib_batch_size_opt) {
    if (lib_batch_size_opt == "auto") {
      batch_size_opt = fit_func->X_train.rows();
    } else {
      int n = fit_func->X_train.rows();
      batch_size_opt = stoi(lib_batch_size_opt);
      if (batch_size_opt > n) {
        //print("[!] Warning: batch size is larger than the number of training examples. Setting it to ", n);
        batch_size_opt = n;
      }
    }
  }

  void apply_feature_selection(int num_feats_to_keep) {
    // check if nothing needs to be done
    if (num_feats_to_keep == -1) {
      return;
    }
    int num_features = 0;
    for(Op * o : terminals) {
      if (o->type() == OpType::otFeat)
        num_features++;
    }
    if (num_features <= num_feats_to_keep)
      return;

    // proceed with feature selection
    Veci indices_to_keep = feature_selection(fit_func->X_train, fit_func->y_train, num_feats_to_keep);
    vector<int> indices_to_remove; indices_to_remove.reserve(terminals.size());
    for(int i = 0; i < terminals.size(); i++) {
      Op * o = terminals[i];
      if (o->type() != OpType::otFeat)
        continue; // ignore constants
      
      auto end = indices_to_keep.data() + indices_to_keep.size();
      if (find(indices_to_keep.data(), end, ((Feat*) o)->id) == end) {
        indices_to_remove.push_back(i);
      }
    }

    // remove those terminals from the search (from back to front not to screw up indexing)
    for(int i = indices_to_remove.size() - 1; i >= 0; i--) {
      int idx = indices_to_remove[i];
      delete terminals[idx];
      terminals.erase(terminals.begin() + idx);
    }

    // gotta update also prob of sampling terminals
    if (lib_tset_probs != "auto") {
      vector<string> prob_str = split_string(lib_tset_probs);
      for(int i = indices_to_remove.size() - 1; i >= 0; i--) {
        int idx = indices_to_remove[i];
        prob_str.erase(prob_str.begin() + idx);
      }
      lib_tset_probs = "";
      for(int i = 0; i < prob_str.size(); i++) {
        lib_tset_probs += prob_str[i];
        if (i < prob_str.size()-1)
          lib_tset_probs += ",";
      }
    }

  }
  

  void reset() {
    for(auto * f : functions) {
      delete f;
    }
    functions.clear();
    for(auto * t : terminals) {
      delete t; 
    }
    terminals.clear();
    if (fit_func)
      delete fit_func;
    if (mse_func)
      delete mse_func;
    fit_func = NULL;
  }

  void read_options(int argc, char** argv) {
    reset();
    cli::Parser parser(argc, argv);

    // budget
    parser.set_optional<int>("pop", "population_size", 1000, "Population size");
    parser.set_optional<int>("g", "generations", 20, "Budget of generations (-1 for disabled)");
    parser.set_optional<int>("t", "time", -1, "Budget of time (-1 for disabled)");
    parser.set_optional<int>("e", "evaluations", -1, "Budget of evaluations (-1 for disabled)");
    parser.set_optional<long>("ne", "node_evaluations", -1, "Budget of node evaluations (-1 for disabled)");
    parser.set_optional<bool>("disable_ims", "disable_ims", false, "Whether to disable the IMS (default is false)");
    // initialization
    parser.set_optional<string>("is", "initialization_strategy", "hh", "Strategy to sample the initial population");
    parser.set_optional<int>("d", "depth", 4, "Maximum depth that the trees can have");
    parser.set_optional<int>("nr_multi_trees", "nr_multi_trees", 1, "Nr of multi trees in individual");
    // problem & representation
    parser.set_optional<string>("ff", "fitness_function", "ac", "Fitness function");
    parser.set_optional<string>("fset", "function_set", "+,-,*,/,sin,cos,log", "Function set");
    parser.set_optional<string>("fset_probs", "function_set_probabilities", "auto", "Probabilities of sampling each element of the function set (same order as fset)");
    parser.set_optional<string>("tset", "terminal_set", "auto", "Terminal set");
    parser.set_optional<string>("tset_probs", "terminal_set_probabilities", "auto", "Probabilities of sampling each element of the function set (same order as tset)");
    parser.set_optional<string>("train", "training_set", "./train.csv", "Path to the training set (needed only if calling as CLI)");
    parser.set_optional<string>("bs", "batch_size", "auto", "Batch size (default is 'auto', i.e., the entire training set)");
    parser.set_optional<string>("compl", "complexity_type", "node_count", "Measure to score the complexity of candidate sotluions (default is node_count)");
    parser.set_optional<float>("rci", "rel_compl_imp", 0.0, "Relative importance of complexity over accuracy to select the final elite (default is 0.0)");
    parser.set_optional<int>("feat_sel", "feature_selection", 10, "Max. number of feature to consider (if -1, all features are considered)");
    // variation
    parser.set_optional<float>("cmp", "coefficient_mutation_probability", 1., "Probability of applying coefficient mutation to a coefficient node");
    parser.set_optional<float>("cmt", "coefficient_mutation_temperature", 0.1, "Temperature of coefficient mutation");
    parser.set_optional<int>("tour", "tournament_size", 2, "Tournament size (if tournament selection is active)");
    parser.set_optional<bool>("nolink", "no_linkage", false, "Disables computing linkage when building the linkage tree FOS, essentially making it random");
    parser.set_optional<bool>("no_large_fos", "no_large_fos", false, "Whether to discard subsets in the FOS with size > half the size of the genotype (default is false)");
    parser.set_optional<bool>("no_univ_fos", "no_univ_fos", false, "Whether to discard univariate subsets in the FOS (default is false)");
    parser.set_optional<bool>("no_univ_exc_leaves_fos", "no_univ_exc_leaves_fos", false, "Whether to discard univariate subsets except for those that refer to leaves in the FOS (default is false)");
    // other
    parser.set_optional<int>("random_state", "random_state", -1, "Random state (seed)");
    parser.set_optional<bool>("verbose", "verbose", false, "Verbose");
    parser.set_optional<bool>("lib", "call_as_lib", false, "Whether the code is called as a library (e.g., from Python)");
    // joe
    parser.set_optional<bool>("use_optim", "use_optim", false, "Whether optimisation is used");
    parser.set_optional<bool>("use_ftol", "use_ftol", false, "Whether ftol is used");
    parser.set_optional<bool>("use_mse_opt", "use_mse_opt", false, "Whether ftol is used");
    parser.set_optional<float>("tol", "tol", 1e-9, "Set tolerance");
    parser.set_optional<bool>("use_clip", "use_clip", false, "Whether gradients are clipped between -1 and 1");
    parser.set_optional<bool>("log", "log", false, "Whether to log");
    parser.set_optional<string>("optimiser_choice", "optimiser_choice", "bfgs", "Selection of optimiser");
    parser.set_optional<int>("opt_per_gen", "opt_per_gen", 1, "Optimise per x gens)");
    parser.set_optional<string>("csv_file", "csv_file", "required.csv", "CSV file that is written to.");
    parser.set_optional<string>("bs_opt", "batch_size_opt", "auto", "Batch size (default is 'auto', i.e., the entire training set)");
    parser.set_optional<bool>("reinject_elite", "reinject_elite", false, "Whether to reinject elites into the new population");
    parser.set_optional<bool>("use_local_search", "use_local_search", false, "Whether local search is used");
    parser.set_optional<bool>("optimise_after", "optimise_after", false, "Whether local search is used");
    parser.set_optional<bool>("add_addition_multiplication", "add_addition_multiplication", false, "Whether addition and multiplication is added to individuals");
    parser.set_optional<bool>("add_any", "add_any", false, "Whether any two function are added to individuals");
    parser.set_optional<bool>("use_max_range", "use_max_range", false, "Whether the max or 10 is used as initalisation range");
    parser.set_optional<bool>("equal_p_coeffs", "equal_p_coeffs", false, "Whether the leafs are sampled with equal probability");
    parser.set_optional<float>("random_accept_p", "random_accept_p", 0.0, "Whether GOM steps are randomly accepted");
  
    // set options
    parser.run_and_exit_if_error();


    // verbose (MUST BE FIRST)
    verbose = parser.get<bool>("verbose");
    if (!verbose) {
      cout.rdbuf(NULL);
    }


    // random_state
    random_state = parser.get<int>("random_state");
    if (random_state >= 0){
      Rng::set_seed(random_state);
      std::srand(random_state);

      //print("random state: ", random_state);
    } else {
      //print("random state: not set");
    }

    // budget
    disable_ims = parser.get<bool>("disable_ims");
    if (disable_ims) {
      pop_size = parser.get<int>("pop");
      //print("pop. size: ",pop_size);
    } else {
      pop_size = 64;
      //print("IMS active");
    }
   
    max_generations = parser.get<int>("g");
    max_time = parser.get<int>("t");
    max_evaluations = parser.get<int>("e");
    max_node_evaluations = parser.get<long>("ne");
    // print("budget: ", 
    //   max_generations > -1 ? max_generations : INF, " generations, ", 
    //   max_time > -1 ? max_time : INF, " time [s], ", 
    //   max_evaluations > -1 ? max_evaluations : INF, " evaluations, ", 
    //   max_node_evaluations > -1 ? max_node_evaluations : INF, " node evaluations" 
    // );


    // initialization
    init_strategy = parser.get<string>("is");
    print("initialization strategy: ", init_strategy);
    max_depth = parser.get<int>("d");
    nr_multi_trees = parser.get<int>("nr_multi_trees");
    add_addition_multiplication = parser.get<bool>("add_addition_multiplication");
    add_any = parser.get<bool>("add_any");

// ============================================================================
//     if(add_addition_multiplication){
//         print(add_addition_multiplication);
//         max_depth = max_depth + 2;
//     }
// ============================================================================
    
    print("max. depth: ", max_depth);
    
    // variation
    cmut_prob = parser.get<float>("cmp");
    cmut_temp = parser.get<float>("cmt");
    //print("coefficient mutation probability: ", cmut_prob, ", temperature: ",cmut_temp);
    tournament_size = parser.get<int>("tour");
    print("tournament size: ", tournament_size);

    no_linkage = parser.get<bool>("nolink");
    no_large_subsets = parser.get<bool>("no_large_fos");
    no_univariate = parser.get<bool>("no_univ_fos");
    no_univariate_except_leaves = parser.get<bool>("no_univ_exc_leaves_fos");
    print("compute linkage: ", no_linkage ? "false" : "true", " (FOS trimming-no large: ",no_large_subsets,", no univ.: ",no_univariate,", no. univ. exc. leaves: ",no_univariate_except_leaves,")");

    // problem
    string fit_func_name = parser.get<string>("ff");
    set_fit_func(fit_func_name);
    //print("fitness function: ", fit_func_name);

    _call_as_lib = parser.get<bool>("lib");
    if (!_call_as_lib) {
      // then it expects a training set
      path_to_training_set = parser.get<string>("train");
      // load up
      if (!exists(path_to_training_set)) {
        throw runtime_error("Training set not found at path "+path_to_training_set);
      }
      Mat Xy = load_csv(path_to_training_set);
      Mat X = remove_column(Xy, Xy.cols()-1);

      Vec y = Xy.col(Xy.cols()-1);
      fit_func->set_Xy(X,y);
      mse_func->set_Xy(X,y);
      if(use_max_range){
          set_max_coeff_range();
        }
    } 
    lib_batch_size = parser.get<string>("bs");
    if (!_call_as_lib) {
      set_batch_size(lib_batch_size);
    }


    lib_batch_size_opt = parser.get<string>("bs_opt");

    if (!_call_as_lib) {
      set_batch_size_opt(lib_batch_size_opt);
      //print("batch size_opt: ", g::batch_size_opt);
    }


    
    use_max_range = parser.get<bool>("use_max_range");
    equal_p_coeffs = parser.get<bool>("equal_p_coeffs");
    random_accept_p = parser.get<float>("random_accept_p");
    //joe
    optimise_after = parser.get<bool>("optimise_after");
    use_clip = parser.get<bool>("use_clip");
    reinject_elite = parser.get<bool>("reinject_elite");
    use_optimiser = parser.get<bool>("use_optim");
    use_local_search = parser.get<bool>("use_local_search");
    use_ftol = parser.get<bool>("use_ftol");
    log = parser.get<bool>("log");
    use_mse_opt = parser.get<bool>("use_mse_opt");
    tol = parser.get<float>("tol");
    opt_per_gen = parser.get<int>("opt_per_gen");

    optimiser_choice = parser.get<string>("optimiser_choice");
    csv_file = parser.get<string>("csv_file");
    //print("optim: ", optimiser_choice, " optimise: ", use_optimiser, " clip: ", use_clip, " reinject elite: ", reinject_elite);


    // representation
    string fset = parser.get<string>("fset");
    set_functions(fset);
    string fset_p = parser.get<string>("fset_probs");
    set_function_probabilities(fset_p);
    //print("function set: ",fset," (probabs: ",fset_p,")");
    
    lib_tset = parser.get<string>("tset");
    lib_feat_sel_number = parser.get<int>("feat_sel");
    lib_tset_probs = parser.get<string>("tset_probs");

    if (!_call_as_lib) {
      set_terminals(lib_tset);
      apply_feature_selection(lib_feat_sel_number);
      set_terminal_probabilities(lib_tset_probs);
      print("terminal set: ",str_terminal_set()," (probs: ",lib_tset_probs, (lib_feat_sel_number > -1 ? ", feat.selection : "+to_string(lib_feat_sel_number) : ""), ")");
    } 


    complexity_type = parser.get<string>("compl");
    rel_compl_importance = parser.get<float>("rci");
    //print("complexity type: ",complexity_type," (rel. importance: ",rel_compl_importance,")");
    // other

    cout << std::setprecision(NUM_PRECISION);

    print("use_max_range " +  std::to_string(use_max_range) 
      + " equal_p_coeffs " +  std::to_string(equal_p_coeffs) +
      + " optimise_after " +  std::to_string(optimise_after) +
      + " use_clip " +  std::to_string(use_clip) +
      + " use_optim " +  std::to_string(use_optimiser) +
      + " use_local_search " +  std::to_string(use_local_search) +
      + " use_ftol " +  std::to_string(use_ftol) +
      + " log " +  std::to_string(log) +
      + " tol " +  std::to_string(tol) +
      + " use_mse_opt " +  std::to_string(use_mse_opt) +
      + " opt_per_gen " +  std::to_string(opt_per_gen) +
      + " add_addition_multiplication " +  std::to_string(add_addition_multiplication)
      + " equal_p_coeffs " +  std::to_string(equal_p_coeffs)
      + " nr multi trees" + std::to_string(nr_multi_trees)
      );


    // float total = 0.;
    // for(int i =0; i<cumul_tset_probs.size();i++){
    //   total += cumul_tset_probs[i];
    // }
    // print(lib_tset_probs);
    // print(to_string(cumul_tset_probs.size()) + " " + to_string(terminals.size()) + " " + to_string(total));
    // throw runtime_error("Unrecognized fitness function: "+fit_func_name);


  }

  void clear_globals() {
    for(auto * o : all_operators) {
      delete o;
    }
    for(auto * f : all_fitness_functions) {
      delete f;
    }
    reset();
  }


}

#endif
