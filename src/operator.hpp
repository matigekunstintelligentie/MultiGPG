#pragma once
#ifndef OPERATOR_H
#define OPERATOR_H

#include <vector>

#include "util.hpp"
#include "rng.hpp"
#include <sstream>

using namespace std;

enum OpType {
  otFun, otFeat, otConst, otPlaceholder, otFunction, otAny
};



template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return std::move(out).str();
}

struct Op {

  virtual ~Op(){

  };

  virtual Op * clone() {
    throw runtime_error("Not implemented clone");  
  }

  virtual int arity() {
    throw runtime_error("Not implemented arity");
  }

    virtual float complexity_kommenda(vector<float> child_complexities) {
        throw runtime_error("Not implemented complexity");
    }

  virtual string sym() {
    throw runtime_error("Not implemented sym");
  }

  virtual string human_repr(vector<string> & args) {
    throw runtime_error("Not implemented human repr");
  }

  virtual string np_repr(vector<string> & args) {
    throw runtime_error("Not implemented");
  }

  virtual string torch_repr(vector<string> & args) {
    throw runtime_error("Not implemented");
  }

  virtual OpType type() {
    throw runtime_error("Not implemented type");
  }

  virtual Vec apply(const Mat & X) {
    throw runtime_error("Not implemented apply");
  }

  virtual pair<Vec, Vec> apply_der(const Mat & X) {
    throw runtime_error("Not implemented apply der");
  }  

  virtual pair<Vec, Vec> apply_der(const Mat & X, Mat & D) {
    throw runtime_error("Not implemented apply der 2");
  }


};

struct Fun : Op {
  OpType type() override {
    return OpType::otFun;
  }

  string _human_repr_binary_between(vector<string> & args) {
    return "(" + args[0] + " " + this->sym() + " " + args[1] + ")";
  }

  string _human_repr_unary_before(vector<string> & args) {
    return this->sym()+ "( " + args[0] + " )";
  }

  string _human_repr_unary_after(vector<string> & args) {
    return "( " + args[0] + " )" + this->sym();
  }

  string _human_repr_common(vector<string> & args) {
    if (arity() == 2) {
      return _human_repr_binary_between(args);
    } else if (arity() == 1) {
      return _human_repr_unary_before(args);
    } else {
      throw runtime_error("Not implemented");
    }
  }

  virtual string human_repr(vector<string> & args) override {
    return _human_repr_common(args);
  }
};

struct Term : Op {
  virtual string human_repr(vector<string> & args) override {
    return sym();
  }
};



struct Add : Fun {

  Op * clone() override {
    return new Add();
  }

  int arity() override {
    return 2;
  }

    float complexity_kommenda(vector<float> child_complexities) override {
        return child_complexities[0] + child_complexities[1] + 1;
    }

  string sym() override {
    return "+";
  }

  Vec apply(const Mat & X) override {
    return X.col(0) + X.col(1);
  }
  
  pair<Vec, Vec> apply_der(const Mat & X, Mat & D) override {
    return make_pair(apply(X), D.col(0) + D.col(1));
  }

  string human_repr(vector<string> & args) override {
    return "(" + args[0]+"+"+args[1] + ")";
  }


  string np_repr(vector<string> & args) override {
    return args[0]+"+"+args[1];
  }

  string torch_repr(vector<string> & args) override {
    return args[0]+"+"+args[1];
  }
};

struct Neg : Fun {

  Op * clone() override {
    return new Neg();
  }

  int arity() override {
    return 1;
  }

  string sym() override {
    return "¬";
  }

    float complexity_kommenda(vector<float> child_complexities) override {
        return child_complexities[0] + 1.;
    }

  Vec apply(const Mat & X) override {
    return -X.col(0);
  }

  pair<Vec, Vec> apply_der(const Mat & X, Mat & D) override {
    return make_pair(apply(X), -D);
  }

  string human_repr(vector<string> & args) override {
    return "(-" + args[0] + ")";
  }

  string np_repr(vector<string> & args) override {
    return "-" + args[0];
  }

  string torch_repr(vector<string> & args) override {
    return "-" + args[0];
  }
};

struct Sub : Fun {

  Op * clone() override {
    return new Sub();
  }

  int arity() override {
    return 2;
  }

    float complexity_kommenda(vector<float> child_complexities) override {
        return child_complexities[0] + child_complexities[1] + 1;
    }

  string sym() override {
    return "-";
  }

  Vec apply(const Mat & X) override {
    return X.col(0)-X.col(1);
  }

  pair<Vec, Vec> apply_der(const Mat & X, Mat & D) override {
    return make_pair(apply(X), D.col(0)-D.col(1));
  }

  string human_repr(vector<string> & args) override {
    return "(" + args[0]+"-"+args[1] + ")";
  }

  string np_repr(vector<string> & args) override {
    return args[0]+"-"+args[1];
  }

  string torch_repr(vector<string> & args) override {
    return args[0]+"-"+args[1];
  }
};

struct Mul : Fun {

  Op * clone() override {
    return new Mul();
  }

  int arity() override {
    return 2;
  }

    float complexity_kommenda(vector<float> child_complexities) override {
        return child_complexities[0] + child_complexities[1] + 2.;
    }

  string sym() override {
    return "*";
  }
  
  Vec apply(const Mat & X) override {
    return X.col(0)*X.col(1);
  }

   pair<Vec, Vec> apply_der(const Mat & X, Mat & D) override {
    return make_pair(apply(X), X.col(1)*D.col(0) + X.col(0)*D.col(1));
  }

  string human_repr(vector<string> & args) override {
    return "(" + args[0]+"*"+args[1] + ")";
  }

  string np_repr(vector<string> & args) override {
    return args[0]+"*"+args[1];
  }

  string torch_repr(vector<string> & args) override {
    return args[0]+"*"+args[1];
  }
};

struct Inv : Fun {

  Op * clone() override {
    return new Inv();
  }

  int arity() override {
    return 1;
  }

  string sym() override {
    return "1/";
  }

    float complexity_kommenda(vector<float> child_complexities) override {
        return child_complexities[0] + child_complexities[1] + 2.;
    }
  
  Vec apply(const Mat & X) override {
    // division by 0 is undefined thus conver to NAN
    Vec denom = X.col(0);
    replace(denom, 0, NAN);
    return 1./denom;
  }

  pair<Vec, Vec> apply_der(const Mat & X, Mat & D) override {
    Vec denom = X.col(0).square();
    replace(denom, 0, NAN);
    return make_pair(apply(X), -D.col(0)/denom);
  }

  string human_repr(vector<string> & args) override {

    return "(1./" + args[0] + ")";
  }  

  string np_repr(vector<string> & args) override {
    return "1./" + args[0];
  }

  string torch_repr(vector<string> & args) override {
    return "1./" + args[0];
  }
};

struct Div : Fun {

  Op * clone() override {
    return new Div();
  }

  int arity() override {
    return 2;
  }

  string sym() override {
    return "/";
  }

    float complexity_kommenda(vector<float> child_complexities) override {
        return child_complexities[0] + child_complexities[1] + 2.;
    }

  Vec apply(const Mat & X) override {
    // division by 0 is undefined thus convert to NAN
    Vec denom = X.col(1);
    replace(denom, 0, NAN);
    return X.col(0)/denom;
  }

  pair<Vec, Vec> apply_der(const Mat & X, Mat & D) override {
    Vec denom = X.col(1);
    replace(denom, 0, NAN);
    return make_pair(apply(X), (X.col(1)*D.col(0) - X.col(0)*D.col(1))/denom.square());
  }

  string human_repr(vector<string> & args) override {
    return "(" + args[0] + "/" + args[1] + ")";
  }

  string np_repr(vector<string> & args) override {
    return args[0] + "/" + args[1];
  }

  string torch_repr(vector<string> & args) override {
    return args[0] + "/" + args[1];
  }
};

struct Sin : Fun {

  Op * clone() override {
    return new Sin();
  }

  int arity() override {
    return 1;
  }

    float complexity_kommenda(vector<float> child_complexities) override {
        return child_complexities[0] + child_complexities[1] + 10.;
    }

  string sym() override {
    return "sin";
  }

  Vec apply(const Mat & X) override {
    return X.sin();
  }

  pair<Vec, Vec> apply_der(const Mat & X, Mat & D) override {
    return make_pair(apply(X), X.cos()*D.col(0));
  }

  string human_repr(vector<string> & args) override {

    return "sin(" + args[0] + ")";
  }

  string np_repr(vector<string> & args) override {
    return "sin(" + args[0] + ")";
  }

  string torch_repr(vector<string> & args) override {
    return "torch.sin(" + args[0] + ")";
  }
};

struct Abs : Fun {

  Op * clone() override {
    return new Abs();
  }

  int arity() override {
    return 1;
  }


    float complexity_kommenda(vector<float> child_complexities) override {
        return child_complexities[0] + 2.;
    }

  string sym() override {
    return "abs";
  }

  Vec apply(const Mat & X) override {
    return X.abs();
  }

  pair<Vec, Vec> apply_der(const Mat & X, Mat & D) override {
    Vec applied = apply(X);
    Vec appliedzero = applied;
    replace(appliedzero, 0, NAN);
    return make_pair(applied, (X.col(0)*D.col(0))/appliedzero);
  }

  string human_repr(vector<string> & args) override {

    return "abs(" + args[0] + ")";
  }

  string np_repr(vector<string> & args) override {
    return "abs(" + args[0] + ")";
  }

  string torch_repr(vector<string> & args) override {
    return "torch.abs(" + args[0] + ")";
  }
};

struct Exp : Fun {

  Op * clone() override {
    return new Exp();
  }

  int arity() override {
    return 1;
  }

    float complexity_kommenda(vector<float> child_complexities) override {
        return child_complexities[0] + 5.;
    }

  string sym() override {
    return "exp";
  }

  Vec apply(const Mat & X) override {
    return X.exp();
  }

  pair<Vec, Vec> apply_der(const Mat & X, Mat & D) override {
    Vec applied = apply(X);
    return make_pair(applied, applied * D.col(0));
  }

  string human_repr(vector<string> & args) override {

    return "exp(" + args[0] + ")";
  }

  string np_repr(vector<string> & args) override {
    return "exp(" + args[0] + ")";
  }

  string torch_repr(vector<string> & args) override {
    return "torch.exp(" + args[0] + ")";
  }
};

struct Pow : Fun {

  Op * clone() override {
    return new Pow();
  }

  int arity() override {
    return 2;
  }

    float complexity_kommenda(vector<float> child_complexities) override {
        return child_complexities[0] + 5.;
    }


    string sym() override {
    return "pow";
  }

  Vec apply(const Mat & X) override {
    return X.col(0).pow(X.col(1));
  }

  pair<Vec, Vec> apply_der(const Mat & X, Mat & D) override {
    return make_pair(apply(X), X.col(0).pow(X.col(1)-1.)*(X.col(0)*D.col(1)*X.col(0).log() + X.col(1)*D.col(0)));
  }

  string human_repr(vector<string> & args) override {

    return "(" + args[0] + "**" + args[1] + ")";
  }

  string np_repr(vector<string> & args) override {
    return args[0] + "**" + args[1];
  }

  string torch_repr(vector<string> & args) override {
    return args[0] + "**" + args[1];
  }
};


struct Max : Fun {

  Op * clone() override {
    return new Max();
  }

  int arity() override {
    return 2;
  }

    float complexity_kommenda(vector<float> child_complexities) override {
        return child_complexities[0] + child_complexities[1] + 2.;
    }


    string sym() override {
    return "max";
  }

  Vec apply(const Mat & X) override {
    return X.col(0).max(X.col(1));
  }

  pair<Vec, Vec> apply_der(const Mat & X, Mat & D) override {
    Vec gt = (X.col(0)>=X.col(1)).cast<float>();
    return make_pair(apply(X), gt*D.col(0) + (1.-gt)*D.col(1));
  }

  string human_repr(vector<string> & args) override {

    return "max("+args[0]+","+args[1]+")";
  }

  string np_repr(vector<string> & args) override {
    return "max(" + args[0] + "," + args[1] + ")";
  }

  string torch_repr(vector<string> & args) override {
    return "torch.maximum(" + args[0] + "," + args[1] + ")";
  }
};

struct Min : Fun {

  Op * clone() override {
    return new Min();
  }

  int arity() override {
    return 2;
  }

    float complexity_kommenda(vector<float> child_complexities) override {
        return child_complexities[0] + child_complexities[1] + 2.;
    }

  string sym() override {
    return "min";
  }

  Vec apply(const Mat & X) override {
    return X.col(0).min(X.col(1));
  }

  pair<Vec, Vec> apply_der(const Mat & X, Mat & D) override {
    Vec gt = (X.col(0)<=X.col(1)).cast<float>();
    return make_pair(apply(X), gt*D.col(0) + (1.-gt)*D.col(1));
  }

  string human_repr(vector<string> & args) override {

    return "min("+args[0]+","+args[1]+")";
  }

  string np_repr(vector<string> & args) override {
    return "min(" + args[0] + "," + args[1] + ")";
  }

  string torch_repr(vector<string> & args) override {
    return "torch.minimum(" + args[0] + "," + args[1] + ")";
  }
};


struct Cos : Fun {

  Op * clone() override {
    return new Cos();
  }

  int arity() override {
    return 1;
  }

    float complexity_kommenda(vector<float> child_complexities) override {
        return child_complexities[0] + 10.;
    }

  string sym() override {
    return "cos";
  }

  Vec apply(const Mat & X) override {
    return X.cos();
  }

  pair<Vec, Vec> apply_der(const Mat & X, Mat & D) override {
    return make_pair(apply(X), D.col(0)*(-X.sin()));
  }

  string human_repr(vector<string> & args) override {

    return "cos("+args[0]+")";
  }

  string np_repr(vector<string> & args) override {
    return "cos(" + args[0] + ")";
  }

  string torch_repr(vector<string> & args) override {
    return "torch.cos(" + args[0] + ")";
  }
};

struct Log : Fun {

  Op * clone() override {
    return new Log();
  }

  int arity() override {
    return 1;
  }

    float complexity_kommenda(vector<float> child_complexities) override {
        return child_complexities[0] + 5.;
    }

  string sym() override {
    return "log";
  }

  Vec apply(const Mat & X) override {
    // Log of x < 0 is undefined and log of 0 is -inf
    return X.col(0).log();
  }

  pair<Vec, Vec> apply_der(const Mat & X, Mat & D) override {
    Vec denom = X.col(0);
    replace(denom, 0, NAN);
    return make_pair(apply(X), D.col(0)/denom);
  }

  string human_repr(vector<string> & args) override {

    return "ln("+args[0]+")";
  }

  string np_repr(vector<string> & args) override {
    return "np.log(" + args[0] + ")";
  }

  string torch_repr(vector<string> & args) override {
    return "torch.log(" + args[0] + ")";
  }
};

struct Sqrt : Fun {

  Op * clone() override {
    return new Sqrt();
  }

  int arity() override {
    return 1;
  }

    float complexity_kommenda(vector<float> child_complexities) override {
        return child_complexities[0] + 5.;
    }

  string sym() override {
    return "sqrt";
  }

  Vec apply(const Mat & X) override {
    // Sqrt of x < 0 is undefined
    return X.col(0).sqrt();
  }

  pair<Vec, Vec> apply_der(const Mat & X, Mat & D) override {
    Vec sqrt = apply(X);
    return make_pair(sqrt, D.col(0)/(2.*sqrt));
  }

  string human_repr(vector<string> & args) override {

    return "sqrt("+args[0]+")";
  }

  string np_repr(vector<string> & args) override {
    return "np.sqrt(" + args[0] + ")";
  }

  string torch_repr(vector<string> & args) override {
    return "torch.sqrt(" + args[0] + ")";
  }
};

struct Square : Fun {

  Op * clone() override {
    return new Square();
  }

  int arity() override {
    return 1;
  }


    float complexity_kommenda(vector<float> child_complexities) override {
        return child_complexities[0] + 5.;
    }

  string sym() override {
    return "**2";
  }

  // string human_repr(vector<string> & args) override {
  //   return _human_repr_unary_after(args);
  // }

  Vec apply(const Mat & X) override {
    Vec x = X.col(0);
    return x.square();
  }

  pair<Vec, Vec> apply_der(const Mat & X, Mat & D) override {
    return make_pair(apply(X), 2.*X*D.col(0));
  }

  string human_repr(vector<string> & args) override {

    return "(" + args[0] + "**2)";
  }

  string np_repr(vector<string> & args) override {
    return "np.square(" + args[0] + ")";
  }

  string torch_repr(vector<string> & args) override {
    return "torch.square(" + args[0] + ")";
  }
};

struct Cube : Fun {

  Op * clone() override {
    return new Cube();
  }

  int arity() override {
    return 1;
  }

  float complexity_kommenda(vector<float> child_complexities) override {
    return child_complexities[0] + 5.;
  }

  string sym() override {
    return "**3";
  }

  // string human_repr(vector<string> & args) override {
  //   return _human_repr_unary_after(args);
  // }

  Vec apply(const Mat & X) override {
    Vec x = X.col(0);
    return x.cube();
  }

  pair<Vec, Vec> apply_der(const Mat & X, Mat & D) override {
    return make_pair(apply(X), 3.*X.square()*D.col(0));
  }

  string human_repr(vector<string> & args) override {

    return "(" + args[0] + "**3)";
  }  

  string np_repr(vector<string> & args) override {
    return args[0] + "**3";
  }

  string torch_repr(vector<string> & args) override {
    return args[0] + "**3";
  }
};

struct Nothing : Fun {
    Op * clone() override {
        return new Nothing();
    }

    int arity() override {
        return 1;
    }

    float complexity_kommenda(vector<float> child_complexities) override {
        return child_complexities[0];
    }

    string sym() override {
        return "nthg";
    }

    // string human_repr(vector<string> & args) override {
    //   return _human_repr_unary_after(args);
    // }

    Vec apply(const Mat & X) override {
        return X;
    }

    pair<Vec, Vec> apply_der(const Mat & X, Mat & D) override {
        return make_pair(apply(X), D);
    }

    string human_repr(vector<string> & args) override {

        return "nthg(" + args[0] + ")";
    }

    string np_repr(vector<string> & args) override {
        return args[0];
    }

    string torch_repr(vector<string> & args) override {
        return args[0];
    }
};


struct Feat : Term {

  int id;
  Feat(int id) {
    this->id = id;
  }

  Op * clone() override {
    return new Feat(this->id);
  }

  int arity() override {
    return 0;
  }

  float complexity_kommenda(vector<float> child_complexities) override {
    return 2.;
  }

  string sym() override {
    return "x_"+to_string(id);
  }

  OpType type() override {
    return OpType::otFeat;
  }

  Vec apply(const Mat & X) override {
    return X.col(id);
  }

  pair<Vec, Vec> apply_der(const Mat & X) override {
    return make_pair(X.col(id), Vec::Constant(X.rows(), 0.));
  }

  string human_repr(vector<string> & args) override {

    return "x_"+to_string(id);
  }

  string np_repr(vector<string> & args) override {
    return "x_"+to_string(id);
  }

  string torch_repr(vector<string> & args) override {
    return "x_"+to_string(id);
  }
};

struct Const : Term {

  float c;
  float d;
  float range;
  Const(float c=NAN, float range=10.) {

    this->c=c;

    this->range=range;
    this->d= static_cast<float>(0);
      if (isnan(c)){
          _sample();
      }
    /*
    if (abs(this->c) < 1e-6) {
      this->c = 0;
    }
    */
  }

  Op * clone() override {
    return new Const(this->c);
  }

  void _sample() {
    this->c = 2*Rng::randu()*this->range - (this->range);
    //roundd(Rng::randu()*10 - 5, NUM_PRECISION);
  }

  int arity() override {
    return 0;
  }

  float complexity_kommenda(vector<float> child_complexities) override{
      return 1.;
  }

  string sym() override {
    if (isnan(c)){
      _sample();
      }
    return to_string_with_precision(c, NUM_PRECISION);
  }

  OpType type() override {
    return OpType::otConst;
  }

  Vec apply(const Mat & X) override {
    if (isnan(c)){
      _sample();
      }
    Vec c_vec = Vec::Constant(X.rows(), c);
    return c_vec;
  }

  pair<Vec, Vec> apply_der(const Mat & X) override {
    return make_pair(Vec::Constant(X.rows(), c), Vec::Constant(X.rows(), d));
  }

  string human_repr(vector<string> & args) override {
    return sym();
  }

  string np_repr(vector<string> & args) override {
    return sym();
  }

  string torch_repr(vector<string> & args) override {
    return sym();
  }
};

struct AnyOp : Term {

    int id;
    AnyOp(int id) {
        this->id = id;
    }

    Op * clone() override {
        return new AnyOp(this->id);
    }

    int arity() override {
        return 0;
    }

    string sym() override {
        return "a_"+to_string(id);
    }

    OpType type() override {
        return OpType::otAny;
    }

    Vec apply(const Mat & X) override {
        Vec c_vec = Vec::Constant(X.rows(), NAN);
        //Vec c_vec = Vec::Random(X.rows());
        return c_vec;
    }

    string human_repr(vector<string> & args) override {

        return "Any_"+to_string(id);
    }

    string np_repr(vector<string> & args) override {
        return "Any_"+to_string(id);
    }

    string torch_repr(vector<string> & args) override {
        return "Any_"+to_string(id);
    }
};

struct FunctionTree : Fun {

    int id;
    FunctionTree(int id) {
        this->id = id;
    }

    Op * clone() override {
        return new FunctionTree(this->id);
    }

    int arity() override {
        return 0;
    }

    string sym() override {
        return "f_" + to_string(id);
    }

    OpType type() override {
        return OpType::otFunction;
    }

    string human_repr(vector<string> & args) override {

        return "Func_" + to_string(id);
    }

    string np_repr(vector<string> & args) override {
        return "Func_" + to_string(id);
    }

    string torch_repr(vector<string> & args) override {
        return "Func_" + to_string(id);
    }
};

struct OutputTree : Term {

    int id;
    OutputTree(int id) {
        this->id = id;
    }

    Op * clone() override {
        return new OutputTree(this->id);
    }

    int arity() override {
        return 0;
    }

    string sym() override {
        return "p_"+to_string(id);
    }

    OpType type() override {
        return OpType::otPlaceholder;
    }

    string human_repr(vector<string> & args) override {

        return "Placeholder_"+to_string(id);
    }

    string np_repr(vector<string> & args) override {
        return "Placeholder_"+to_string(id);
    }

    string torch_repr(vector<string> & args) override {
        return "Placeholder_"+to_string(id);
    }
};

#endif