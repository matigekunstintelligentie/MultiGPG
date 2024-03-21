#ifndef ELITISTARCHIVE_H
#define ELITISTARCHIVE_H

#include "util.hpp"
#include "rng.hpp"

struct ElitistArchive{
    vector<Individual*> SO_archive;
    vector<Individual*> MO_archive;
    Fitness * fit_func;
    bool improved_this_gen = false;
    int generations_without_improvement = 0;
    bool accept_diversity = false;

    Mat X_train;

    ~ElitistArchive(){
        for(auto ind: MO_archive){
            ind->clear();
        }
        for(auto ind: SO_archive){
            ind->clear();
        }
    }

    void set_X(Mat & X){
        X_train = X;
    }

    Individual * ReturnCopyRandomMOMember() {
        int index = Rng::randu() * MO_archive.size();
        Individual *copy = MO_archive[index]->clone();
        return copy;
    }

    Individual * ReturnCopySOMember(int idx) {
        Individual *copy = SO_archive[idx]->clone();
        return copy;
    }

    bool nondominated(Individual* ind){
        bool solution_is_dominated = false;
        bool identical_objectives_already_exist;
        bool diversity_added = false;

        if (MO_archive.empty()) {
            return true;
        }
        for (size_t i = 0; i < MO_archive.size(); i++) {
            // check domination
            solution_is_dominated = dominates(MO_archive[i], ind);
            if (solution_is_dominated) {
                break;
            }

            identical_objectives_already_exist = true;
            for (size_t j = 0; j < 2; j++) {
                if (ind->fitness[j] != MO_archive[i]->fitness[j]) {
                    identical_objectives_already_exist = false;
                    break;
                }
            }
            if (identical_objectives_already_exist) {
                if (diversityAdded(ind, i)) {
                    diversity_added = true;
                }
                break;
            }
        }
        return (!solution_is_dominated && !identical_objectives_already_exist) || (diversity_added);
    }

    bool dominates(Individual * ind1, Individual * ind2){
        bool strictly_better_somewhere = false;
        for(int i = 0; i<2; i++){
            if(ind1->fitness[i] < ind2->fitness[i]){
                strictly_better_somewhere = true;
            }
            else if(ind1->fitness[i] > ind2->fitness[i]){
                return false;
            }
        }
        return strictly_better_somewhere;
    }

    bool diversityAdded(Individual* individual, int idx){
        Vec diff = individual->get_output(X_train) - MO_archive[idx]->get_output(X_train);
        if(diff.mean()==0){
            return false;
        }
        else{
            return true;
        }
    }

    void initSOArchive(vector<Individual*> population){
        SO_archive = vector<Individual*>(2, nullptr);
        for(Individual *ind: population){
            updateSOArchive(ind);
        }
    }

    void initMOArchive(vector<Individual*> population){
        for(Individual *ind: population){
            updateMOArchive(ind);
        }
    }

    void updateSOArchive(Individual * individual){
        for(int i=0; i<2; i++){
            if(SO_archive[i] == nullptr){
                Individual *new_individual = individual->clone();
                SO_archive[i] = new_individual;
            }
            else if(SO_archive[i]->fitness[i] > individual->fitness[i]){
                SO_archive[i]->clear();
                SO_archive[i] = nullptr;
                Individual *new_individual = individual->clone();
                SO_archive[i] = new_individual;
            }
        }
    }

    void updateMOArchive(Individual * individual){
        bool solution_is_dominated = false;
        bool diversity_added = false;
        bool identical_objectives_already_exist;

        for(int i = 0; i<MO_archive.size(); i++){
            // Check domination
            solution_is_dominated = dominates(MO_archive[i], individual);
            // If solution is dominated then do not add
            if(solution_is_dominated){
                break;
            }

            identical_objectives_already_exist = true;
            for(int j=0; j<2; j++){
                if(individual->fitness[j] != MO_archive[i]->fitness[j]){
                    identical_objectives_already_exist = false;
                    break;
                }
            }

            if(identical_objectives_already_exist && accept_diversity){
                if (diversityAdded(individual, i)) {
                    diversity_added = true;
                    MO_archive[i]->clear();
                    MO_archive[i] = nullptr;
                }
                break;
            }

            if(dominates(individual, MO_archive[i])){
                MO_archive[i]->clear();
                MO_archive[i] = nullptr;
            }
        }
        MO_archive.erase(std::remove_if(MO_archive.begin(), MO_archive.end(), [](Individual *ind){return ind== nullptr;}), MO_archive.end());

        if ((!solution_is_dominated && !identical_objectives_already_exist) || (diversity_added)) {
            Individual *new_individual = individual->clone();
            improved_this_gen = true;
            MO_archive.push_back(new_individual);
        }
    }
};

#endif