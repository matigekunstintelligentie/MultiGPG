#ifndef TESTS_H
#define TESTS_H

#include "myeig.hpp"
#include "util.hpp"
#include "node.hpp"
#include "operator.hpp"
#include "fitness.hpp"
#include "variation.hpp"

using namespace std;
using namespace myeig;

struct Test {

  void run_all() {
      // num_nodes();
// ============================================================================
//     depth();
//     subtree();
//     gen_tree();
//     operators();
//     node_output();
//     fitness();
//     converge();
//     math();
// ============================================================================
  }

  void num_nodes(){
      Node * first_tree = new Node(new Sin());
      Node * z = new Node(new Cos());
      first_tree->append(z);
      Node * any = new Node(new AnyOp(0));
      z->append(any);

      Node * x = new Node(new FunctionTree(0));
      Node * y = new Node(new Const(0.5));
      x->append(y);

      Individual * ind = new Individual();
      ind->trees.push_back(first_tree);
      ind->trees.push_back(x);

      Mat matA(2, 2);
      matA << 1, 2, 3, 4;

      Vec output = ind->get_output(matA);

      print(output(0));

      print(ind->human_repr(true));
      print(ind->get_num_nodes(true));

      assert(1==2);
  }

// ============================================================================
//   Node * _generate_mock_tree() {
//     // Builds x_0 * (x_1 + x_1) aka [* x_0 + x_1 x_1]
//     Node * add_node = new Node(new Add());
//     Node * mul_node = new Node(new Mul());
//     Node * feat0_node = new Node(new Feat(0));
//     Node * feat1_node = new Node(new Feat(1));
//     Node * feat1_node2 = feat1_node->clone();
// 
//     add_node->append(feat1_node);
//     add_node->append(feat1_node2);
// 
//     mul_node->append(feat0_node);
//     mul_node->append(add_node);
// 
//     return mul_node;
//   }
// 
//   void depth() {
//     Node n1(new Add());
//     Node n2(new Add());
//     Node n3(new Add());
// 
//     n3.parent = &n2;
//     n2.parent = &n1;
// 
//     assert(n1.depth() == 0);
//     assert(n2.depth() == 1);
//     assert(n3.depth() == 2);
//   }
// 
//   void subtree() {
//     Node n1(new Add()), n2(new Add()), n3(new Add()), n4(new Add()), n5(new Add());
//     n1.fitness = 1;
//     n2.fitness = 2;
//     n3.fitness = 3;
//     n4.fitness = 4;
//     n5.fitness = 5;
// 
//     n1.children.push_back(&n2);
//     n2.children.push_back(&n3);
//     n2.children.push_back(&n5);
//     n1.children.push_back(&n4);
// 
//     auto subtree = n1.subtree();
//     string collected_fitnesses = "";
//     for(Node * n : subtree) {
//       collected_fitnesses += to_string((int)n->fitness);
//     }
//     
//     assert(collected_fitnesses == "12354");
// 
//   }
// 
//   void gen_tree() {
//     vector<Op*> functions = {new Add(), new Sub(), new Mul()};
//     vector<Op*> terminals = {new Feat(0), new Feat(1)};
//     for(int height = 10; height >= 0; height--){
//       // create full trees, check their height is correct
//       for(int trial=0; trial < 10; trial++){
//         auto * t = _grow_tree_recursive(2, height, height, -1, 0.0);
//         assert(t->height() == height);
//         t->clear();
//       }
//     }
//     for(auto * op : functions)
//       delete op;
//     for(auto * op : terminals)
//       delete op;
//   }
// 
//   void operators() {
// 
//     // Generic ref
//     Op * op;
// 
//     // Toy data
//     Mat X(3,2);
//     X << 1, 2,
//          3, 4,
//          5, 6;
//     Vec expected(3);
//     Vec result(3);
// 
//     // Add
//     op = new Add();
//     assert(op->sym() == "+");
//     expected << 3, 7, 11;
//     result = op->apply(X);
//     assert(result.isApprox(expected));
//     delete op;
// 
//     // Sub
//     op = new Sub();
//     expected << -1, -1, -1;
//     result = op->apply(X);
//     delete op;
// 
//     // Neg
//     op = new Neg();
//     expected << -1, -3, -5;
//     Mat temp = X.col(0);
//     result = op->apply(temp);
//     delete op;
//     
//   }
// 
//   void node_output() {
//     // Toy data
//     Mat X(3,2);
//     X << 1, 2,
//          3, 4,
//          5, 6;
//     Vec expected(3);
//     Vec result(3);
// 
//     Node * mock_tree = _generate_mock_tree();
//     expected << 4, 24, 60;
//     result = mock_tree->get_output(X);
//     mock_tree->clear();
// 
//     assert(result.isApprox(expected));
//   }
// 
//   void fitness() {
//     auto * mock_tree = _generate_mock_tree();
// 
//     Mat X(3,2);
//     X << 1, 2,
//          3, 4,
//          5, 6;
//     Vec y(3);
//     y << 1, 0, 1;
// 
//     Vec out = mock_tree->get_output(X);
//     // out = [4, 24, 60]
// 
//     // mae = mean(|1-4|,|0-24|,|1-60|)
//     float expected = (3+24+59)/3.0;
// 
//     Fitness * f = new MAEFitness();
//     float res = f->get_fitness(mock_tree, X, y);
// 
//     assert(res == expected);
//     delete f;
//     mock_tree->clear();
//   }
// 
//   void converge() {
//     Evolution * e = new Evolution(0);
// 
//     // test a converged population
//     vector<Node*> population; 
//     for(int i = 0; i < 10; i++){
//       population.push_back(_generate_mock_tree());
//     }
//     //TODO: fix this test
//     //assert(e->converged(population));
// 
//     // test a non-converged population
//     auto nodes = population[3]->subtree();
//     delete nodes[0]->op;
//     nodes[0]->op = new Feat(9);
//     //assert(!e->converged(population));
// 
//     // cleanup
//     e->clear_population(population);
//     delete e;
//   }
// 
//   void math() {
//     Vec y(5);
//     y << 2, 4, 6, 8, 10;
//
//     assert( corr(x, y) == 1.0 );
//
//     y = -1 * y;
//     assert( corr(x, y) == -1.0 );
//
//     y << 0, 0, 0, 0, 0;
//     assert( corr(x, y) == 0.0 );
//
//     // nan & infs
//     assert(isnan(NAN));
//     assert(INF > 99999999.9);
//     assert(NINF < -9999999.9);
//
//     // round
//     assert(roundd(0.1, 0)==0);
//     assert(roundd(0.5, 0)==1);
//     assert(roundd(0.0004, 7)==(float)0.0004);
//     assert(roundd(0.0004, 3)==0);
//     assert(roundd(0.0027, 3)==(float)0.003);
//
//     // sort_order
//     Vec a(5);
//     a << 10., .1, 3., 2., 11.;
//     Veci order_of_a(5);
//     Veci rank_of_a(5);
//     order_of_a << 1, 3, 2, 0, 4;
//     rank_of_a << 3, 0, 2, 1, 4;
//     Veci o = sort_order(a);
//     assert(order_of_a.isApprox(o));
//     Veci r = ranking(a);
//     assert(rank_of_a.isApprox(r));
//   }
// ============================================================================
//     // correlation
//     Vec x(5); 
//     x << 1, 2, 3, 4, 5;
//     Vec y(5);
//     y << 2, 4, 6, 8, 10;
// 
//     assert( corr(x, y) == 1.0 );
// 
//     y = -1 * y;
//     assert( corr(x, y) == -1.0 );
// 
//     y << 0, 0, 0, 0, 0;
//     assert( corr(x, y) == 0.0 );
// 
//     // nan & infs
//     assert(isnan(NAN));
//     assert(INF > 99999999.9);
//     assert(NINF < -9999999.9);
// 
//     // round
//     assert(roundd(0.1, 0)==0);
//     assert(roundd(0.5, 0)==1);
//     assert(roundd(0.0004, 7)==(float)0.0004);
//     assert(roundd(0.0004, 3)==0);
//     assert(roundd(0.0027, 3)==(float)0.003);
// 
//     // sort_order
//     Vec a(5);
//     a << 10., .1, 3., 2., 11.;
//     Veci order_of_a(5);
//     Veci rank_of_a(5);
//     order_of_a << 1, 3, 2, 0, 4;
//     rank_of_a << 3, 0, 2, 1, 4;
//     Veci o = sort_order(a);
//     assert(order_of_a.isApprox(o));
//     Veci r = ranking(a);
//     assert(rank_of_a.isApprox(r));
//   }
// ============================================================================

};


#endif