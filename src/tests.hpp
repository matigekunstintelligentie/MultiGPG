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

  void run_all(){
      complexity_kommenda();
      //funcs();
    // num_nodes();
// ============================================================================
     height();
    // subtree();
//    is_intron();
//     gen_tree();
//     operators();
//     node_output();
//     fitness();
//     converge();
//     math();
// ============================================================================
  }


//    Individual * joe = new Individual();
//    Node * sin = new Node(new Sin());
//    Node * plus = new Node(new Add());
//    Node * any0 = new Node(new AnyOp(0));
//    Node * any1 = new Node(new AnyOp(1));
//
//
//
//    Node * plus0 = new Node(new Add());
//
//    Node * plus1 = new Node(new Add());
//    Node * plus2 = new Node(new Add());
//
//
//
//    Node * plus3 = new Node(new Add());
//    Node * plus4 = new Node(new Add());
//
//
//
//    Node * plus5 = new Node(new Add());
//    Node * plus6 = new Node(new Add());
//
//
//
//    Node * A7 = new Node(new FunctionTree(0));
//    Node * A8 = new Node(new FunctionTree(0));
//
//
//
//    Node * A9 = new Node(new FunctionTree(0));
//    Node * A10 = new Node(new FunctionTree(0));
//
//
//
//    Node * A11 = new Node(new FunctionTree(0));
//    Node * A12 = new Node(new FunctionTree(0));
//    Node * A13 = new Node(new FunctionTree(0));
//    Node * A14 = new Node(new FunctionTree(0));
//
//
//
//    Node * f0 = new Node(new Feat(0));
//    Node * f1 = new Node(new Feat(1));
//    Node * f2 = new Node(new Feat(0));
//    Node * f3 = new Node(new Feat(2));
//    Node * f4 = new Node(new Feat(0));
//    Node * f5 = new Node(new Feat(3));
//    Node * f6 = new Node(new Feat(0));
//    Node * f7 = new Node(new Feat(4));
//    Node * f8 = new Node(new Feat(0));
//    Node * f9 = new Node(new Feat(5));
//    Node * f10 = new Node(new Feat(0));
//    Node * f11 = new Node(new Feat(6));
//    Node * f12 = new Node(new Feat(0));
//    Node * f13 = new Node(new Feat(7));
//    Node * f14 = new Node(new Feat(0));
//    Node * f15 = new Node(new Feat(8));
//
//    A7->append(f0);
//    A7->append(f1);
//    A8->append(f2);
//    A8->append(f3);
//    A9->append(f4);
//    A9->append(f5);
//    A10->append(f6);
//    A10->append(f7);
//    A11->append(f8);
//    A11->append(f9);
//    A12->append(f10);
//    A12->append(f11);
//    A13->append(f12);
//    A13->append(f13);
//    A14->append(f14);
//    A14->append(f15);
//
//    plus6->append(A13);
//    plus6->append(A14);
//    plus5->append(A11);
//    plus5->append(A12);
//    plus4->append(A9);
//    plus4->append(A10);
//    plus3->append(A7);
//    plus3->append(A8);
//    plus2->append(plus5);
//    plus2->append(plus6);
//    plus1->append(plus3);
//    plus1->append(plus4);
//    plus0->append(plus1);
//    plus0->append(plus2);
//
//    plus->append(any0);
//    plus->append(any1);
//    sin->append(plus);
//
//    joe->trees.push_back(sin);
//    joe->trees.push_back(plus0);
//
//    print(joe->human_repr(true));
//    print(to_string(g::fit_func->get_fitness_MO(joe)[0]));

    void funcs(){
        Node * first_tree = new Node(new Sin());
        Node * any = new Node(new AnyOp(0));
        first_tree->append(any);

        Node * second_tree = new Node(new Cos());
        Node * x = new Node(new FunctionTree(0));
        Node * y = new Node(new Const(0.5));
        x->append(y);
        second_tree->append(x);

        Node * x2 = new Node(new FunctionTree(1));
        Node * y2 = new Node(new Const(0.5));
        x2->append(y2);

        Individual * ind = new Individual();
        ind->trees.push_back(first_tree);
        ind->trees.push_back(second_tree);
        ind->trees.push_back(x2);

//        print(ind->human_repr(true));
//        print(ind->get_num_nodes(true));
//
//        assert(4==2);
    }

    void is_intron(){
        Node * first = new Node(new Const(5));
        Node * second = new Node(new Const(6));

        Individual * ind = new Individual();
        ind->trees.push_back(first);
        ind->trees.push_back(second);

        assert(ind->is_intron(first)==true);
        assert(ind->is_intron(second)==false);

        ind->clear();
    }

    void complexity_kommenda(){
//        Node * e = new Node(new Exp());
//        Node * sin = new Node(new Sin());
//        Node * sqrt1 = new Node(new Sqrt());
//        Node * x = new Node(new Feat(0));
//
//        sqrt1->append(x);
//        sin->append(sqrt1);
//        e->append(sin);
//
//        Individual * ind1 = new Individual();
//        ind1->trees.push_back(e);
//
//        print(ind1->get_complexity_kommenda());
//
//        assert(abs(ind1->get_complexity_kommenda()-65536.)<1e-6);

        Node * p1 = new Node(new Add());
        Node * p2 = new Node(new Add());
        Node * c5 = new Node(new Const(5.));
        Node * x2 = new Node(new Feat(0));
        Node * mul1 =  new Node(new Mul());
        Node * mul2 = new Node(new Mul());
        Node * sqrt = new Node(new Square());

        sqrt->append(x2);
        mul1->append(sqrt);
        mul1->append(c5);

        mul2->append(x2);
        mul2->append(c5);

        p2->append(mul1);
        p2->append(mul2);
        p1->append(p2);
        p1->append(c5);

        Individual * ind2 = new Individual();
        ind2->trees.push_back(p1);

        print(ind2->get_complexity_kommenda());
        assert(abs(ind2->get_complexity_kommenda()-17.)<1e-6);
    }

  void num_nodes(){
      Node * first_tree = new Node(new Sin());
      Node * z = new Node(new Mul());
      first_tree->append(z);
      Node * any = new Node(new AnyOp(0));
      z->append(any);
      z->append(any);

      Node * x = new Node(new FunctionTree(0));
      Node * y = new Node(new Const(0.5));
      Node * y2 = new Node(new Const(0.6));
      x->append(y);
      y->append(y2);

      Individual * ind = new Individual();
      ind->trees.push_back(first_tree);
      ind->trees.push_back(x);

      assert(ind->get_num_nodes(true)==4);
      assert(ind->get_num_nodes(false)==6);
      assert(ind->get_num_nodes(true, true)==3);
  }

    void height(){
        Node * first_tree = new Node(new Sin());
        Node * z = new Node(new Mul());
        Node * z2 = new Node(new Mul());
        Node * any = new Node(new AnyOp(0));
        z->append(any);
        z2->append(any);
        z2->append(any);
        z->append(z2);
        first_tree->append(z);

        Node * x = new Node(new FunctionTree(0));
        Node * y = new Node(new Const(0.5));
        Node * y2 = new Node(new Const(0.6));
        x->append(y);
        y->append(y2);

        Individual * ind = new Individual();
        ind->trees.push_back(first_tree);
        ind->trees.push_back(x);

        print(ind->human_repr(true));
        print(ind->get_height());
        assert(ind->get_height()==4);
    }

  void subtree(){
      Node * first_tree = new Node(new Sin());
      Node * z = new Node(new Cos());
      first_tree->append(z);
      Node * any = new Node(new AnyOp(0));
      z->append(any);

      Node * x = new Node(new FunctionTree(0));
      Node * y = new Node(new Const(0.5));
      Node * y2 = new Node(new Const(0.6));
      x->append(y);
      y->append(y2);

      Individual * ind = new Individual();
      ind->trees.push_back(first_tree);
      ind->trees.push_back(x);

      vector<Node*> nodes = ind->subtree(true);

      assert(nodes.size()==4);

      nodes = ind->subtree(false);

      assert(nodes.size()==3);

      ind->clear();
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
//
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