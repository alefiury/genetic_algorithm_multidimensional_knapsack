import random
import logging
from typing import List, Tuple
import matplotlib.pyplot as plt

import tqdm
import wandb
import numpy as np
from pyparsing import Optional

from utils.utils import formatter_single

# Logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=formatter_single.FORMATTER)
# np.random.seed(1024)

class GeneticAlgorithm:
    def __init__(
        self,
        num_generations: int,
        population_size: int,
        crossover_rate: float,
        mutation_rate: float,
        round_decimals: int,
        num_variables: int,
        num_constraints: int,
        coeficients: List[int],
        constraints: List[int],
        max_weigths: List[int],
        selection: str,
        k: Optional,
        repair: str,
        num_parents_last_gen: int
    ):

        self.num_generations = num_generations
        self.population_size = population_size
        self.num_variables = num_variables
        self.num_constraints = num_constraints
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.round_decimals = round_decimals
        self.coeficients = coeficients
        self.constraints = constraints
        self.max_weigths = max_weigths
        self.selection = selection
        self.k = k
        self.repair = repair
        self.num_parents_last_gen = num_parents_last_gen


    def init_individuals(self) -> None:
        """
        Creates an initial population based on a discrete uniform distribution
        """
        # Sample integers from a discrete uniform distribution, in the range [0, 2)
        self.individuals = np.random.randint(0, 2, (self.population_size, self.num_variables))


    def bit2solution(self) -> List[float]:
        """
        Converts a bit string in its solution representation
        """
        solutions = np.repeat(self.coeficients[None,:], self.population_size, axis=0)*self.individuals

        return solutions


    def get_fitness(self) -> List[float]:
        """
        Gets fitness values for all individuals
        """
        solutions = self.bit2solution()

        return np.sum(solutions, axis=-1)


    def get_weights(self) -> List[float]:
        """
        Gets the constraints for each individual, for the whole population
        """
        individuals_rep = np.matrix.repeat(self.individuals, self.num_constraints, axis=0)
        constraints_rep = np.matrix.repeat(self.constraints[None, :], self.population_size, axis=0).reshape((self.population_size*self.num_constraints, self.num_variables))
        weights = individuals_rep*constraints_rep
        weights = weights.reshape((self.population_size, self.num_constraints, self.num_variables))

        return weights


    def get_individual_weights(self, individual: List[int]) -> List[float]:
        """
        Get constraints for one individual
        """
        individuals_rep = np.matrix.repeat(individual[None, :], self.num_constraints, axis=0)
        weights = individuals_rep*self.constraints

        return np.sum(weights, axis=-1)


    def get_total_weights(self) -> List[float]:
        return np.sum(self.get_weights(), axis=-1)


    def repair_benefit_cost(self) -> List[float]:
        new_individuals = []
        weights = self.get_total_weights()
        max_weigth_rep = np.matrix.repeat(self.max_weigths[None, :], self.population_size, axis=0)
        mask = weights > max_weigth_rep

        costs = np.array(self.constraints)
        benefits = np.array(self.coeficients)
        benefits_rep = np.matrix.repeat(benefits[None, :], self.num_constraints, axis=0)
        benefit_cost_rate = benefits_rep/costs
        benefit_cost_rate_mean = np.mean(benefit_cost_rate, axis=0)
        benefit_cost_rate_mean_ind = np.argsort(benefit_cost_rate_mean)

        for individual, ind_mask in zip(self.individuals, mask):
            mask = ind_mask
            while np.count_nonzero(mask):
                individual_idxs = np.nonzero(individual)
                wrost_item = benefit_cost_rate_mean_ind[np.in1d(benefit_cost_rate_mean_ind, individual_idxs)][0]
                individual[wrost_item] = 0

                individual_weights = self.get_individual_weights(individual)
                mask = individual_weights > self.max_weigths

            new_individuals.append(individual)
        self.individuals = np.array(new_individuals)


    def repair_random(self) -> None:
        new_individuals = []
        weights = self.get_total_weights()
        max_weigth_rep = np.matrix.repeat(self.max_weigths[None, :], self.population_size, axis=0)
        mask = weights > max_weigth_rep

        for individual, ind_mask in zip(self.individuals, mask):
            mask = ind_mask
            while np.count_nonzero(mask):
                choosen_items = np.argwhere(individual == 1).flatten()
                rand_idx = np.random.choice(choosen_items)
                individual[rand_idx] = 0

                individual_weights = self.get_individual_weights(individual)
                mask = individual_weights > self.max_weigths

            new_individuals.append(individual)

        self.individuals = np.array(new_individuals)


    def crossover(self, individual1: List[str], individual2: List[str]) -> List[str]:
        """
        Performs the crossover operation onto 2 individuals
        """
        sep = np.random.randint(0, self.num_variables-1)
        offspring = np.concatenate((individual1[:sep], individual2[sep:]), axis=None)

        return offspring


    def mutation(self, individual: List[int], mutation_rate: float) -> List[str]:
        """
        Invert a bit based on a certain probability
        """
        individual_copy = np.copy(individual)
        probs = np.random.uniform(0, 1, individual.shape[0])
        mask = probs < mutation_rate

        individual_copy[mask] = np.logical_not(individual_copy[mask]).astype(int)

        return individual_copy


    def roulette_wheel_selection(self) -> int:
        """
        Selects an individual based on the roulette wheel strategy
        """
        fitness = self.get_fitness()
        fitness_cumulative_sum = np.cumsum(fitness)
        rand_num = np.random.uniform(0, np.max(fitness_cumulative_sum))

        selected_individual = np.argmax(fitness_cumulative_sum >= rand_num)

        return selected_individual


    def tournament_selection(self, k: int) -> List[int]:
        choosen_individuals_indxs = np.random.choice(np.arange(self.population_size), k)
        choosen_individuals = self.individuals[choosen_individuals_indxs]
        solutions = np.repeat(self.coeficients[None,:], k, axis=0)*choosen_individuals
        solutions = np.sum(solutions, axis=-1)

        return choosen_individuals_indxs[np.argmax(solutions)]


    def iterate(self) -> Tuple:

        self.init_individuals()
        if self.repair == "random":
            self.repair_random()
        elif self.repair == "cost_benefit" or self.repair == "hybrid":
            self.repair_benefit_cost()
        else:
            log.info("This repair technique does not exist...")
            exit()
        fitness = self.get_fitness()

        for gen in tqdm.tqdm(range(self.num_generations)):
            # Gets best individual in its real representation (x, y)
            idx_best_fitness = np.argmax(fitness)
            idx_best_fitness_sorted = np.flip(np.argsort(fitness))

            log.info(f"Generation: {gen+1}/{self.num_generations} | Avg Fitness: {np.round(np.mean(fitness), self.round_decimals)} | Best Fitness: {np.round(np.max(fitness), self.round_decimals)}")
            log.info(f"Best Cromossom: {''.join(list(map(str, self.individuals[idx_best_fitness].tolist())))}")

            selected_individuals = []
            offsprings = []
            mutated_offsprings = []

            # Selection
            for _ in range(self.population_size):
                if self.selection == "tournament":
                    si = self.tournament_selection(self.k)
                if self.selection == "roulette":
                    si = self.roulette_wheel_selection()
                selected_individuals.append(si)

            # Crossover
            mates = np.random.choice(selected_individuals, self.population_size, replace=False)
            for selected_individual, mate in zip(selected_individuals, mates):
                if np.random.uniform(0, 1) < self.crossover_rate:
                    offsprings.append(self.crossover(self.individuals[selected_individual], self.individuals[mate]))
                else:
                    offsprings.append(self.individuals[selected_individual])

            # Mutation
            for offspring in offsprings:
                mutated_offsprings.append(self.mutation(offspring, self.mutation_rate))

            # Remove M offsprings and adds M parents
            random.shuffle(mutated_offsprings)
            mutated_offsprings = mutated_offsprings[self.num_parents_last_gen:]
            mutated_offsprings = np.concatenate((mutated_offsprings, self.individuals[idx_best_fitness_sorted[:self.num_parents_last_gen]]), axis=0)

            self.individuals = np.array(mutated_offsprings)
            if self.repair == "random":
                self.repair_random()
            elif self.repair == "cost_benefit":
                self.repair_benefit_cost()
            elif self.repair == "hybrid":
                if gen > self.num_generations*0.5:
                    self.repair_benefit_cost()
                else:
                    self.repair_random()

            fitness = self.get_fitness()

            wandb.log(
                {
                    "generation": gen,
                    "best_fitness": np.max(fitness),
                    "avg_fitness": np.mean(fitness),
                    "best_individual": ''.join(list(map(str, self.individuals[idx_best_fitness].tolist())))
                }
            )

        return np.round(np.max(fitness), self.round_decimals), np.round(np.mean(fitness), self.round_decimals), ''.join(list(map(str, self.individuals[idx_best_fitness].tolist())))