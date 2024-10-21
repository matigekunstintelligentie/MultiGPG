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
#include "rng.hpp"
#include "elitistarchive.hpp"

using namespace std;
using namespace myeig;

#include <filesystem>
namespace fs = std::filesystem;

namespace g {

  // ALL operators
  vector<Op*> all_operators = {
    new Add(), new Sub(), new Neg(), new Mul(), new Div(), new Inv(), 
    new Square(), new Sqrt(), new Cube(),
    new Sin(), new Cos(), new Nothing(),
    new Log(), new Pow(), new Max(), new Min(), new Exp(), new Abs()
  };

  // ALL fitness functions 
  vector<Fitness*> all_fitness_functions = {
    new MSEFitness(), new LSMSEFitness()
  };

  // budget
  int pop_size;
  int max_generations;
  int max_time;
  int max_evals;
  int max_non_improve;

  bool MO_mode = false;
  bool full_mode = false;

  bool use_adf = false;
  bool use_aro = false;

  bool use_GA = false;
  bool use_GP = false;
  bool drift = false;
  bool koza = false;

  int n_clusters = 7;
  int nr_objs = 2;

    float cmut_eps;
    float cmut_prob;
    float cmut_temp;

  // logging
  string csv_file;
  string csv_file_pop;
  bool log = false;
  bool log_pop = false;
  bool log_front = false;
  bool accept_diversity = false;

  // Optimisation choices
  // Optimiser specific
  string optimiser_choice;
  bool use_optimiser=false;
  bool use_clip=false;
  float tol;
  bool use_ftol=false;
  bool use_mse_opt=false;
  int lm_max_fev = 10;

  int opt_per_gen;

  // GOMEA choices
  int nr_multi_trees;

  bool balanced = false;
  bool k2 = false;
  float donor_fraction = 2.;

  // coefficients and range
  float range = 10.;
  bool use_max_range=false;
  bool equal_p_coeffs=false;
  int max_coeffs;
  bool discount_size=false;
  bool  change_second_obj = false;

  bool remove_duplicates = false;
  string replacement_strategy;

  // representation
  int max_depth;
  string init_strategy;
  vector<Op*> functions;
  vector<Op*> terminals;

  string lib_tset; // used when `fit` is called when using as lib
  string lib_tset_probs; // used when `fit` is called when using as lib

  int lib_feat_sel_number = -1; // used when `fit` is called when using as lib

  // problem
  Fitness * fit_func = NULL;
  Fitness * mse_func = NULL;

  ElitistArchive * ea = new ElitistArchive();


  string path_to_training_set;
  string path_to_validation_set;
  string lib_batch_size; // used when `fit` is called when using as lib
  int batch_size;
  string lib_batch_size_opt;
  int batch_size_opt;

  // variation
  int max_init_attempts = 10000;

  // selection
  int tournament_size;

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
          int n_coeffs=fit_func->X_train.cols();
          if(g::max_coeffs>-1){
              n_coeffs = g::max_coeffs;
          }
        for(int i = 0; i < n_coeffs; i++) {
          terminals.push_back(new Const(NAN,range));
        }
      }
      else{
          if(g::max_coeffs!=0) {
              terminals.push_back(new Const(NAN, range));
          }
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
            terminals.push_back(new Const(c, range));
          }
        } catch(std::invalid_argument const& ex) {
          throw runtime_error("Unrecognized terminal: "+sym);
        }
      }
    }
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
      batch_size_opt = mse_func->X_train.rows();
    } else {
      int n = mse_func->X_train.rows();
      batch_size_opt = stoi(lib_batch_size_opt);
      if (batch_size_opt > n) {
        //print("[!] Warning: batch size is larger than the number of training examples. Setting it to ", n);
        batch_size_opt = n;
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
    parser.set_optional<int>("pop", "population_size", 4096, "Population size");
    parser.set_optional<int>("g", "generations", 20, "Budget of generations (-1 for disabled)");
    parser.set_optional<int>("t", "time", -1, "Budget of time (-1 for disabled)");
    parser.set_optional<int>("e", "evals", -1, "Budget of evals (-1 for disabled)");
    parser.set_optional<int>("max_non_improve", "max_non_improve", -1, "Budget of non_improves (-1 for disabled)");

    // initialization
    parser.set_optional<string>("is", "initialization_strategy", "hh", "Strategy to sample the initial population");
    parser.set_optional<int>("d", "depth", 4, "Maximum depth that the trees can have");
    parser.set_optional<int>("nr_multi_trees", "nr_multi_trees", 2, "Nr of multi trees in individual");
    // problem & representation
    parser.set_optional<string>("ff", "fitness_function", "lsmse", "Fitness function");
    parser.set_optional<string>("fset", "function_set", "+,-,*,/,sin,cos,log", "Function set");
    parser.set_optional<string>("tset", "terminal_set", "auto", "Terminal set");
    parser.set_optional<string>("train", "training_set", "./train.csv", "Path to the training set (needed only if calling as CLI)");
    parser.set_optional<string>("val", "validation_set", "./val.csv", "Path to the validation set (needed only if calling as CLI)");
    parser.set_optional<string>("bs", "batch_size", "auto", "Batch size (default is 'auto', i.e., the entire training set)");
    // variation
    parser.set_optional<float>("cmp", "coefficient_mutation_probability", 1., "Probability of applying coefficient mutation to a coefficient node");
    parser.set_optional<float>("cmt", "coefficient_mutation_temperature", 0.1, "Temperature of coefficient mutation");
    parser.set_optional<int>("tour", "tournament_size", 4, "Tournament size (if tournament selection is active)");
    // other
    parser.set_optional<int>("random_state", "random_state", -1, "Random state (seed)");
    parser.set_optional<bool>("verbose", "verbose", false, "Verbose");
    parser.set_optional<bool>("lib", "call_as_lib", false, "Whether the code is called as a library (e.g., from Python)");
    // optimisation
    parser.set_optional<bool>("use_optim", "use_optimiser", false, "Whether optimisation is used");
    parser.set_optional<bool>("use_ftol", "use_ftol", false, "Whether ftol is used");
    parser.set_optional<bool>("use_mse_opt", "use_mse_opt", false, "Whether the mse is optimised is used");
    parser.set_optional<float>("tol", "tol", 1e-9, "Set tolerance");
    parser.set_optional<bool>("use_clip", "use_clip", false, "Whether gradients are clipped between -1 and 1");
    parser.set_optional<string>("bs_opt", "batch_size_opt", "auto", "Batch size (default is 'auto', i.e., the entire training set)");
    //gomea
    parser.set_optional<int>("opt_per_gen", "opt_per_gen", 1, "Optimise per x gens)");
    // logging
    parser.set_optional<bool>("log", "log", false, "Whether to log");
    parser.set_optional<bool>("log_front", "log_front", false, "Whether to log nondom front");

    parser.set_optional<bool>("log_pop", "log_pop", false, "Whether to log pop");
    parser.set_optional<string>("csv_file", "csv_file", "required.csv", "CSV file that is written to.");
    parser.set_optional<string>("csv_file_pop", "csv_file_pop", "required_pop.csv", "CSV file that pop cluster information is written to.");

    parser.set_optional<bool>("remove_duplicates", "remove_duplicates", false, "Whether duplicates are removed.");
    parser.set_optional<string>("replacement_strategy", "replacement_strategy", "mutate", "Replacement strategy for when removing duplicates.");


    // coefficients and range
    parser.set_optional<bool>("use_max_range", "use_max_range", false, "Whether the max or 10 is used as initalisation range");
    parser.set_optional<bool>("equal_p_coeffs", "equal_p_coeffs", false, "Whether the leafs are sampled with equal probability");
    parser.set_optional<int>("max_coeffs", "max_coeffs", -1, "Maximum number of Coefficients");

    parser.set_optional<bool>("MO_mode", "MO_mode", false, "Whether Multi objective mode is activated");
    parser.set_optional<bool>("use_adf", "use_adf", false, "Whether Automatically Defined Functions are used");
    parser.set_optional<bool>("use_aro", "use_aro", false, "Whether Automatically Re-used outputs are used");
    parser.set_optional<bool>("use_GA", "use_GA", false, "Whether GA or GOMEA is used");
    parser.set_optional<bool>("use_GP", "use_GP", false, "Whether GP or GOMEA is used");
    parser.set_optional<bool>("koza", "koza", false, "Whether Koza style HADFs are used.");

    parser.set_optional<bool>("full_mode", "full_mode", false, "Whether all trees are initialized full.");
    parser.set_optional<bool>("drift", "drift", false, "Whether intron changes are kept.");
    parser.set_optional<bool>("discount_size", "discount_size", false, "Whether the model size is discounted for re-use");
    parser.set_optional<bool>("change_second_obj", "change_second_obj", false, "Whether the second obj is complexity or arbitrary complexity");
    parser.set_optional<bool>("balanced", "balanced", false, "Whether balanced k-leaders is used");
    parser.set_optional<bool>("k2", "k2", false, "Whether balanced k-2-leaders is used");
    parser.set_optional<bool>("accept_diversity", "accept_diversity", false, "Whether non-dominated, but equal objective solution are accepted into the MO-archive");
    parser.set_optional<float>("donor_fraction", "donor_fraction", 2., "What fraction of the closest full population is used as donor population");
    parser.set_optional<float>("rci", "rci", 1., "Relative complexity importance");
    parser.set_optional<int>("n_clusters", "n_clusters", 7, "Number of clusters");

      parser.set_optional<int>("nr_objs", "nr_objs", 2, "Number of objs, hardcoded to be mse, size, complexity measure");

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

      print("random state: ", random_state);
    } else {
      print("random state: not set");
    }

    // budget
    pop_size = parser.get<int>("pop");
    print("pop. size: ",pop_size);

    max_evals = parser.get<int>("e");
    max_generations = parser.get<int>("g");
    max_time = parser.get<int>("t");
    max_non_improve = parser.get<int>("max_non_improve");
    print("budget: ",
       max_generations > -1 ? max_generations : INF, " generations, ",
       max_time > -1 ? max_time : INF, " time [s], "
    );


    // initialization
    init_strategy = parser.get<string>("is");
    print("initialization strategy: ", init_strategy);
    max_depth = parser.get<int>("d");
    nr_multi_trees = parser.get<int>("nr_multi_trees");

    for(int i =0;i<nr_multi_trees-1;i++){
        if(use_adf || use_aro) {
            all_operators.push_back(new FunctionTree(i));
        }
    }
    if(use_adf) {
        all_operators.push_back(new AnyOp(0));
        all_operators.push_back(new AnyOp(1));
    }

    print("max. depth: ", max_depth);
    
    // variation
    cmut_prob = parser.get<float>("cmp");
    cmut_temp = parser.get<float>("cmt");
    print("coefficient mutation probability: ", cmut_prob, ", temperature: ",cmut_temp);
    tournament_size = parser.get<int>("tour");
    print("tournament size: ", tournament_size);

    // problem
    string fit_func_name = parser.get<string>("ff");
    set_fit_func(fit_func_name);
    print("fitness function: ", fit_func_name);

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
        path_to_validation_set = parser.get<string>("val");
        // load up
        if (!exists(path_to_validation_set)) {
            throw runtime_error("Training set not found at path "+path_to_validation_set);
        }
        Mat Xy_val = load_csv(path_to_validation_set);
        Mat X_val = remove_column(Xy_val, Xy_val.cols()-1);

        Vec y_val = Xy_val.col(Xy_val.cols()-1);


        fit_func->set_Xy(X_val,y_val, "val");
        mse_func->set_Xy(X_val,y_val, "val");

        ea->set_X(fit_func->X_train);
        ea->fit_func = fit_func;
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


    MO_mode = parser.get<bool>("MO_mode");
    use_adf = parser.get<bool>("use_adf");
    use_aro = parser.get<bool>("use_aro");
    use_GA = parser.get<bool>("use_GA");
    use_GP = parser.get<bool>("use_GP");
    koza = parser.get<bool>("koza");
    drift = parser.get<bool>("drift");

    discount_size = parser.get<bool>("discount_size");
    change_second_obj = parser.get<bool>("change_second_obj");
    balanced = parser.get<bool>("balanced");
    k2 = parser.get<bool>("k2");
    accept_diversity = parser.get<bool>("accept_diversity");
    ea->accept_diversity = accept_diversity;
    donor_fraction = parser.get<float>("donor_fraction");

    n_clusters = parser.get<int>("n_clusters");
    nr_objs = parser.get<int>("nr_objs");
    ea->nr_objs = nr_objs;
    use_max_range = parser.get<bool>("use_max_range");
    equal_p_coeffs = parser.get<bool>("equal_p_coeffs");
    max_coeffs = parser.get<int>("max_coeffs");
    //joe
    remove_duplicates = parser.get<bool>("remove_duplicates");
    replacement_strategy = parser.get<string>("replacement_strategy");
    use_clip = parser.get<bool>("use_clip");
    use_optimiser = parser.get<bool>("use_optim");
    use_ftol = parser.get<bool>("use_ftol");
    log = parser.get<bool>("log");
    log_front = parser.get<bool>("log_front");
    log_pop = parser.get<bool>("log_pop");
    use_mse_opt = parser.get<bool>("use_mse_opt");
    tol = parser.get<float>("tol");
    opt_per_gen = parser.get<int>("opt_per_gen");

    csv_file = parser.get<string>("csv_file");
    csv_file_pop = parser.get<string>("csv_file_pop");
    //print("optim: ", optimiser_choice, " optimise: ", use_optimiser, " clip: ", use_clip, " reinject elite: ", reinject_elite);




    // representation
    string fset = parser.get<string>("fset");
    set_functions(fset);
    //print("function set: ",fset," (probabs: ",fset_p,")");
    
    lib_tset = parser.get<string>("tset");

    if (!_call_as_lib) {
      set_terminals(lib_tset);

    }

      if(g::use_adf) {
          for(int i =0;i<int(1);i++) {
              g::terminals.push_back(new AnyOp(0));
              g::terminals.push_back(new AnyOp(1));
          }
      }


      for(int i = 0; i<g::nr_multi_trees - 1;i++){
          if(g::use_adf) {
              g::functions.push_back(new FunctionTree(i));
          }
          if(g::use_aro){
              g::terminals.push_back(new OutputTree(i));
          }
      }
    print("terminal set: ",str_terminal_set());

    cout << std::setprecision(NUM_PRECISION);

    print("use_max_range " +  std::to_string(use_max_range) 
      + " equal_p_coeffs " +  std::to_string(equal_p_coeffs) +
      + " max_coeffs " +  std::to_string(max_coeffs) +
      + " use_clip " +  std::to_string(use_clip) +
      + " use_optim " +  std::to_string(use_optimiser) +
      + " use_ftol " +  std::to_string(use_ftol) +
      + " log " +  std::to_string(log) +
      + " log pop " +  std::to_string(log_pop) +
      + " log front " +  std::to_string(log_front) +
      + " tol " +  std::to_string(tol) +
      + " use_mse_opt " +  std::to_string(use_mse_opt) +
      + " opt_per_gen " +  std::to_string(opt_per_gen) +
      + " equal_p_coeffs " +  std::to_string(equal_p_coeffs)
      + " nr multi trees " + std::to_string(nr_multi_trees)
      + " MO mode " + std::to_string(MO_mode)
      + " n clusters " + std::to_string(n_clusters)
      + " GA mode " + std::to_string(use_GA)
      + " GP mode " + std::to_string(use_GP)
      + " Koza mode " + std::to_string(koza)
      + "drift " + std::to_string(drift)
      + " use aro " + std::to_string(use_aro)
      + " use adf " + std::to_string(use_adf)
      + " balanced " + std::to_string(balanced)
      + " k2 " + std::to_string(k2)
      + " donor_fraction " + std::to_string(donor_fraction)
      + " accept_diversity " + std::to_string(accept_diversity)
      );


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
