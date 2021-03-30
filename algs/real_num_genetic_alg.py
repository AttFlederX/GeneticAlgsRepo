import numpy as np
import matplotlib.pyplot as plt

class RealNumberGeneticAlgorithm:
    '''
        Modified implementation of a genetic algorithm for finding the local minimum of a function, 
        using real number individuals

        Problem parameters:
            a, b: double             - interval on which to find the local minimum,
            f: (double) -> double    - function to be minimized,
            prec: uint               - precision in positive powers of 10,

            n_p - size of population,
            n_t - size of tournament,
            p_e - percentage for the elitism model,
            p_m - probability of mutation,
            p_c - probability of crossover,
            t_max_i - maximum number of unchanging generations,
            t_max - maximum number of total generations,
            b_dep - degree of dependency on # of iteration
    '''
    def __init__(self,
                a,
                b,
                f,
                prec,

                n_p = 100,
                n_t = 2,
                p_e = 0.1,
                p_m = 0.01,
                p_c = 0.8,
                t_max_i = 10,
                t_max = 1000,
                b_dep = 5):

        assert a < b, 'Invalid interval'
        assert prec > 1, 'Invalid precision'

        assert n_t >= 2 and n_p >= n_t, 'Invalid population params' 
        assert 0 <= p_e <= 1 and 0 <= p_m <= 1 and 0 <= p_c <= 1, 'Invalid probability params' 
        assert t_max_i > 2 and t_max >= t_max_i, 'Invalid stopping conditions' 
        assert b_dep > 1, 'Invalid degree of dependency'

        self._a = a
        self._b = b
        self._f = f
        self._prec = prec

        self._n_p = n_p 
        self._n_t = n_t 
        self._p_e = p_e
        self._p_m = p_m 
        self._p_c = p_c 
        self._t_max_i = t_max_i 
        self._t_max = t_max
        self._b_dep = b_dep


    ## Algorithm functions

    def __generate_population(self):
        '''
            Generates a random population in the form of real numbers from a to b
        '''
        pop = list()

        for i in range(self._n_p):
            # generate a real number within the interval
            newIndiv = self._a + (self._b - self._a) * np.random.random()

            pop.append(newIndiv)

        return pop


    def __fitness(self, v):
        '''
            Fitness function for an individual (less => more fit)
        '''
        return self._f(v)

    def __population_fitness(self, pop):
        '''
            For a given population, returns its best fitness & the individual which has it
        '''
        # find the smallest (i.e. best) fitness in a population
        min = np.min(list(map(lambda v: self.__fitness(v), pop)))
        # find the individial with the smallest fitness
        minIndiv = next(filter(lambda v: self.__fitness(v) <= min, pop))

        return min, minIndiv


    ## Generational operators

    def __select(self, pop):
        '''
            Perform a population selection using the elitism & tournament methods

                - Add top p_e percent of individuals straight to the new population
                - Select n_t individuals from the population
                - Add the most fit individual to the new population
                - Repeat until the new population's size is n_p

                (individuals are not required to be unique)
        '''
        # elitism model
        selectedPop = list(pop)
        selectedPop.sort(key=lambda v: self.__fitness(v))
        selectedPop = selectedPop[:int(self._n_p * self._p_e)]

        tournamentSelection = list()

        while len(selectedPop) < self._n_p:
            tournamentSelection.clear()

            # select n_t random individuals
            for i in range(self._n_t):
                randIdx = np.random.randint(self._n_p)
                tournamentSelection.append(pop[randIdx])

            # add the most fit individual into new population
            minFit = np.min(list(map(lambda v: self.__fitness(v), tournamentSelection)))
            selectedPop.append(next(filter(lambda v: self.__fitness(v) <= minFit, tournamentSelection)))

        return selectedPop

    def __mutate(self, pop, t):
        '''
            Perform mutations with individuals from the population

                - Select one random individual
                - With p_m probability, perform a non-uniform mutation on it
                - Add to the new population, regardless of whether the mutation occured
                - Repeat until the new population's size is n_p
        '''

        mutatedPop = list()

        while len(mutatedPop) < self._n_p:
            # pick a random individual
            randIdx = np.random.randint(self._n_p)
            randIndiv = pop[randIdx]

            # with p_m probability, perform a non-uniform mutation
            if np.random.random() <= self._p_m:
                # this whole process is meant to limit how much mutation actually modifies the individual
                # as the number of iterations increases
                # if we were to just pick a random number within the interval, the individual could stray away from
                # the solution in late generations of the algorithm, when it is obviously unwanted
                delta = np.random.random()
                r_t = 1 - (delta ** ((1 - t/self._t_max) ** self._b_dep))

                lbm = max([self._a, randIndiv - (self._b - self._a)*r_t])
                ubm = min([self._b, randIndiv + (self._b - self._a)*r_t])

                mutatedPop.append(lbm + (ubm - lbm)*np.random.random())
            else:
                # put the individual into the new population without change
                mutatedPop.append(randIndiv)

        return mutatedPop

    def __crossover(self, pop):
        '''
            Performs crossovers between random pairs of individuals

                - Pick two random individuals from the population (v_i, v_j)
                - With p_c probability:
                    - Assume v_i < v_j
                    - Pick two random real number within [v_i, v_j]
                    - Add those two to the new population
                - Otherwise, add the two individuals without change
        '''
        crossoverPop = list()

        while len(crossoverPop) < self._n_p:
            # pick two random individuals
            v_i = pop[np.random.randint(self._n_p)]
            v_j = pop[np.random.randint(self._n_p)]

            if np.random.random() <= self._p_c:
                # make sure that v_i < v_j
                if v_i >= v_j:
                    v_i, v_j = v_j, v_i

                # create new individuals & add them to the population
                v_i_new = v_i + (v_j - v_i)*np.random.random()
                v_j_new = v_i + (v_j - v_i)*np.random.random()

                crossoverPop.append(v_i_new)
                crossoverPop.append(v_j_new)
            else:
                # add without change
                crossoverPop.append(v_i)
                crossoverPop.append(v_j)

        return crossoverPop

        
    ## Main entry point

    def solve(self, verbose_mode=False, graphing_mode=False):
        # generate initial population
        pop = self.__generate_population()

        # initialize iteration counter
        t = 0
        # unchanging iteration counter
        t_i = 0
        # fitness stack
        popFitnessStack = []

        # keep track of the best result
        bestFitness = np.infty
        bestFitnessIndiv = None
        
        # begin iterating
        while t < self._t_max:
            # perform generational changes
            pop = self.__select(pop)
            pop = self.__mutate(pop, t)
            pop = self.__crossover(pop)

            # determine best fitness in a population & add it to the stack
            popFitnessVal, bestIndiv = self.__population_fitness(pop)
            popFitnessStack.append(popFitnessVal)

            # update best fitness
            if popFitnessVal < bestFitness:
                bestFitness = popFitnessVal
                bestFitnessIndiv = bestIndiv

            if verbose_mode:
                print(f'Generation #{t}: best fitness {popFitnessVal} at x = {bestIndiv}')

            if t > 1:
                # if the best fitness difference is small, increment the unchanged iteration counter,
                # else, reset it
                if abs(popFitnessStack[t] - popFitnessStack[t - 1]) < 10 ** -self._prec:
                    t_i += 1
                else:
                    t_i = 0

            # if max unchanged iterations reached, stop
            if t_i >= self._t_max_i:
                break

            t += 1

        if graphing_mode:
            plt.plot(popFitnessStack)
            plt.show()

        # return best fitness of final population
        return bestFitness, bestFitnessIndiv, t


