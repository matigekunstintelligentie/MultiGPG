#include <iostream>

#include "globals.hpp"
#include "myeig.hpp"
#include "util.hpp"
#include "evolution.hpp"
#include "ims.hpp"
#include "individual.hpp"
#include "node.hpp"
#include "tests.hpp"

using namespace myeig;

int main(int argc, char** argv){
  g::read_options(argc, argv);

  auto t = Test();
  t.run_all();

//  Individual * ind = generate_individuals(4, "hh", 2);
//
//  Individual * ind2 = ind->clone();
////
//  ind->clear();
//  ind2->clear();
//  print("");
  auto start_time = tick();
  //auto evo = Evolution();
  //evo.run();
  auto * ims = new IMS();
  ims->run();
  delete ims;
  print("Runtime: ",tock(start_time),"s");
//
  g::clear_globals();

}