#ifndef ELITISTARCHIVE_H
#define ELITISTARCHIVE_H

//#include "individual.hpp"
#include "globals.hpp"

struct ElitistArchive{
    vector<Individual*> SO_archive;
    vector<Individual*> MO_archive;

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
        Vec diff = individual->get_output(g::fit_func->X_train, individual->trees) - MO_archive[idx]->get_output(g::fit_func->X_train, individual->trees);
//        if(diff.mean()==0){
//            return false;
//        }
//        else{
            return true;
//        }
    }

    void initSOArchive(vector<Individual*> population){
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
            solution_is_dominated = dominates(MO_archive[i], individual);
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
            if(identical_objectives_already_exist){
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
            //g::fit_func->get_fitness_MO(new_individual);
            MO_archive.push_back(new_individual);
        }
    }
};

#endif