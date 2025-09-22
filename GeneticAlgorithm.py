import math

import numpy as np
import torch

class GeneticAlgorithm:
    def __init__(self):
        self.generations = 0
        self.best_fitnesses = []
        self.max_prev_fitness = 0
        self.pop = 0
        self.pop_size = 0
        self.gene_length = 0


    def do_stuff(self, pop, fitness):
        self.pop = pop
        number_of_mutations = int(pop.shape[1] * 0.2)

        print(pop.shape, number_of_mutations, self.generations)

        self.best_fitnesses.append(np.max(fitness))
        self.pop_size = pop.shape[0]
        self.gene_length = pop.shape[1]

        #print('GA best fitness:', self.best_fitnesses)
        # print("Generation : ", gen, " Best fitness previous gen: ", max(fitness))
        #print(self.pop)

        best_to_keep = int(pop.shape[0] * 0.1) # keep best 10%
        if best_to_keep < 2:
            best_to_keep = 2

        parents = self.select_mating_pool(fitness.copy(), best_to_keep)


        offspring_crossover = self.crossover(parents)
        offspring_mutation = self.mutation(offspring_crossover, num_mutations=number_of_mutations)
        # Creating the new population based on the parents and offspring.
        self.pop[0:parents.shape[0], :] = parents
        self.pop[parents.shape[0]:, :] = offspring_mutation
        shuffle_indices = np.arange(self.pop.shape[0])
        np.random.shuffle(shuffle_indices)
        self.pop = self.pop[shuffle_indices] # randomize

        parent_indices = np.where(shuffle_indices < parents.shape[0])[0]

        self.generations += 1
        #self.save_to_file()

        if self.generations % 10 == 0:
            self.save_best(parents)

        return self.pop, parent_indices

    # print(self.pop[numpy.where(fitness == max(fitness))[0]])

    def select_mating_pool(self, fitness, num_parents):
        parents = np.empty((num_parents, self.pop.shape[1]), dtype=np.float32)

        for parent_num in range(num_parents):
            max_fitness_idx = np.where(fitness == np.max(fitness))
            max_fitness_idx = max_fitness_idx[0][0]
            parents[parent_num, :] = self.pop[max_fitness_idx, :]
            fitness[max_fitness_idx] = -999999999
        # print(parents)
        return parents

    def crossover(self, parents):
        offspring_size = [self.pop_size - parents.shape[0], self.gene_length]
        offspring = np.empty(offspring_size)

        for k in range(offspring_size[0]):
            parents_idx = np.random.choice(parents.shape[0], replace=False, size=2)

            # per il crossover prendo un set di geni dal genitore 1 e riempio gli altri geni con quelli del genitore 2,
            genes = np.random.choice(parents.shape[1], replace=False, size=2)

            startGene = min(genes)
            endGene = max(genes)

            offspring[k, startGene:endGene] = parents[parents_idx[0], startGene:endGene]
            offspring[k, 0:startGene] = parents[parents_idx[1], 0:startGene]
            offspring[k, endGene:] = parents[parents_idx[1], endGene:]

        # print(offspring)
        return offspring

    def mutation(self, offspring_crossover, num_mutations=1):
        # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
        for idx in range(offspring_crossover.shape[0]):  # loop offspring
            mutation_gene_indexes = np.random.choice(offspring_crossover.shape[1], replace=False, size=num_mutations)
            for gene_idx in mutation_gene_indexes:
                offspring_crossover[idx, gene_idx] += 0.1*np.random.uniform(-1, 1)

        return offspring_crossover



    def save_best(self, parents):
        print('saved best weights')
        np.save('ga_best_weights.npy', parents)