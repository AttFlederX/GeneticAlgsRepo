from common import F_norm
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import sort

class MultiVariableGeneticAlgorithm:
    '''
        Multi-variable implementation of a genetic algorithm for solving a system of non-linear equations
        (or any equations, really)

        Problem parameters:
            a, b: double[]              - intervals on which to find the solution,
            f: ((double[]) -> double)[] - system of equations,
            prec: uint                  - precision in positive powers of 10,

            n_p - size of population,
            p_e - percentage for the elitism model,
            p_m - probability of mutation,
            r - r-model parameter,
            t_max_i - maximum number of unchanging generations,
            t_max - maximum number of total generations
    '''
    def __init__(self,
                a,
                b,
                f,
                prec,

                n_p = 100,
                p_e = 0.1,
                p_m = 0.05,
                r = 0.2,
                t_max_i = 10,
                t_max = 1000):

        assert prec > 1, 'Invalid precision'

        assert n_p >= 2, 'Invalid population params' 
        assert 0 <= p_e <= 1 and 0 <= p_m <= 1 and 0 <= r <= 1, 'Invalid probability params' 
        assert t_max_i > 2 and t_max >= t_max_i, 'Invalid stopping conditions' 

        self._a = a
        self._b = b
        self._f = f
        self._prec = prec

        self._n_p = n_p 
        self._p_e = p_e
        self._p_m = p_m 
        self._r = r
        self._t_max_i = t_max_i 
        self._t_max = t_max

        # system dimension
        self._dim = len(f)
        assert len(a) == self._dim and len(b) == self._dim, 'Invalid num of intervals'

        # sizes of individuals
        self._n = []
        
        # size of individual bitword
        self._m = 0


    ## Algorithm functions

    def __find_size_of_individual(self, i):
        '''
            Finds the length of the bit word required for the given precision, 
            by bitshifting the overall number of points on an interval
        '''
        numOfPoints = int((self._b[i] - self._a[i]) * (10 ** self._prec))
        numOfBits = 0

        while numOfPoints > 0:
            numOfPoints = int(numOfPoints / 2) # equivalent to shifting 1 bit to the right
            numOfBits += 1

        return numOfBits

    def __generate_population(self):
        '''
            Generates a random population in the form of string bitwords
        '''
        pop = list()

        for i in range(self._n_p):
            newIndiv = ''

            for n in self._n:
                # generate a zero-padded bitword of size n
                newIndivValue = f"{np.random.randint(2 ** n):b}"
                newIndivValue = newIndivValue.zfill(n)

                newIndiv += newIndivValue

            pop.append(newIndiv)

        return pop


    def __binary_to_digit(self, v, i):
        '''
            Converts a bitword into a position within the interval
        '''
        vInt = int(v, 2) # convert the bitword into a decimal integer
        return self._a[i] + ( ((self._b[i] - self._a[i]) / (2**self._n[i] - 1)) * vInt )

    def __individual_to_double_array(self, v):
        '''
            Converts a bitword individual into an array of real numbers
        '''
        vDoubleArr = np.zeros(self._dim)
        vIdx = 0
    
        for i in range(self._dim):
            # the individual is made up of concatenated bitword strings of each variable, 
            # so we just separate them using size values in n, and convert each one into a number 
            # within the respective domain

            subBitWord = v[vIdx:vIdx+self._n[i]]
            vDoubleArr[i] = self.__binary_to_digit(subBitWord, i)

            vIdx += self._n[i]

        return vDoubleArr


    def __fitness(self, v):
        '''
            Fitness function for an individual (closer to zero => more fit)
        '''
        # f returns an array, so we return its norm to get a 1-D measure of the fitness
        return F_norm(self._f, self.__individual_to_double_array(v))

    def __zero_adjusted_fitness(self, v):
        '''
            Returns a guaranteed non-zero fitness value for finding reciprocals
        '''
        fit = self.__fitness(v)
        if fit != 0:
            return fit
        else:
            return 10.0 ** -16

    def __population_fitness(self, pop):
        '''
            For a given population, returns its best fitness & the individual which has it
        '''
        # find the smallest (i.e. best) fitness in a population
        min = np.min(list(map(lambda v: self.__fitness(v), pop)))
        # find the individial with the smallest fitness
        minIndiv = next(filter(lambda v: self.__fitness(v) <= min, pop))

        return min, self.__individual_to_double_array(minIndiv)


    ## Generational operators

    def __select(self, pop):
        '''
            Perform a population selection using the elitism & roulette methods

                - Add top p_e percent of individuals straight to the new population
                - Get an array of ratios of individual reciprocal fitnesses to total reciprocal fitness
                - Build an array of probability intervals corresponding to each individual's fitness
                - Pick a random number & find the probability range which it falls into
                (the better the fitness, the larger its range, hence the random number is more likely to fall into it)
                - Add the corresponding individual to the new population
                - Repeat until the new population has n_p individuals

                (individuals are not required to be unique)
        '''
        # elitism model
        selectedPop = list(pop)
        selectedPop.sort(key=lambda v: self.__fitness(v))

        # keep a copy of the sorted population for the roulette method
        sortedPop = list(selectedPop)
        sortedPop.reverse()

        selectedPop = selectedPop[:int(self._n_p * self._p_e)]

        # roulette method

        # save total zero-adjusted fitnesses
        zeroAdjustedFitnesses = list(map(lambda v: self.__zero_adjusted_fitness(v), sortedPop))
        # calculate the total fitness of the population
        # since less is better, sum the reciprocals instead of actual values (and hope that fitness is not 0)
        totalFitness = sum(list(map(lambda zaf: (1/zaf), zeroAdjustedFitnesses)))

        # get an array of ratios of fitness values to total fitness
        fitnessProbabilities = list(map(lambda zaf: (1/zaf) / totalFitness, zeroAdjustedFitnesses))

        # build an array of probability ranges
        fitnessRanges = []
        pos = 0

        for prob in fitnessProbabilities:
            fitnessRanges.append([pos, pos + prob])
            pos += prob

        while len(selectedPop) < self._n_p:
            x = np.random.random()

            # pick a random probability range
            # since better fitnesses have larger ranges, their corresponding individuals are more likely to be picked
            for i in range(self._n_p):
                if fitnessRanges[i][0] <= x <= fitnessRanges[i][1]:
                    selectedPop.append(sortedPop[i])
                    break

        return selectedPop

    def __mutate(self, pop):
        '''
            Perform mutations with individuals from the population

                - Select one random individual
                - With p_m probability, flip one random bit in its bitword
                - Add to the new population, regardless of whether the mutation occured
                - Repeat until the new population's size is n_p
        '''

        mutatedPop = list()

        while len(mutatedPop) < self._n_p:
            # pick a random individual
            randIdx = np.random.randint(self._n_p)
            randIndiv = pop[randIdx]

            # flip a random bit with _p_m probability
            if np.random.random() <= self._p_m:
                # split it into char array
                splitRandIndiv = list(randIndiv)

                randBitIdx = np.random.randint(self._m)

                # flip a random bit
                if splitRandIndiv[randBitIdx] == '1':
                    splitRandIndiv[randBitIdx] = '0'
                else:
                    splitRandIndiv[randBitIdx] = '1'

                # join the individual back into a string & put it in the new population
                mutatedPop.append(''.join(splitRandIndiv))
            else:
                # put the individual into the new population without change
                mutatedPop.append(randIndiv)

        return mutatedPop

    def __crossover(self, pop):
        '''
            Performs crossovers between random pairs of individuals using the r-model

                - Pick two random individuals from the top r percent of the population
                - With p_c probability:
                    - Randomly select the split position
                    - Split both individuals in two at the split position
                    - Add an individual comprised of the first part of the first individual and a 
                    second part of the second individual to the new population
                    - Add an individual comprised of the first part of the second individual and a 
                    second part of the first individual to the new population
                - Otherwise, add the two individuals without change
        '''
        crossoverPop = list()
        sortedPop = sorted(list(pop), key=lambda v: self.__fitness(v))

        # get r percent of the population
        rOfPop = int(self._n_p * self._r)

        # only crossover the top p percent of the population
        while len(crossoverPop) < rOfPop:
            # pick two random individuals
            v_i = sortedPop[np.random.randint(rOfPop)]
            v_j = sortedPop[np.random.randint(rOfPop)]

            # pick a split point
            splitPoint = np.random.randint(self._m)

            # split the individuals
            v_i_firstPart = v_i[:splitPoint]
            v_i_secondPart = v_i[splitPoint:]

            v_j_firstPart = v_j[:splitPoint]
            v_j_secondPart = v_j[splitPoint:]

            # put the merged ones into the new population
            crossoverPop.append(f"{v_i_firstPart}{v_j_secondPart}")
            crossoverPop.append(f"{v_j_firstPart}{v_i_secondPart}")

        # add the rest of the individuals randomly, excluding the bottom p percent
        while len(crossoverPop) < self._n_p:
            crossoverPop.append(sortedPop[np.random.randint(self._n_p - rOfPop)])

        return crossoverPop

        
    ## Main entry point

    def solve(self, verbose_mode=False, graphing_mode=False):
        # find bitword length
        for i in range(self._dim):
            assert self._a[i] < self._b[i], 'Invalid interval'
            self._n.append(self.__find_size_of_individual(i))

            self._m += self._n[i]

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
            pop = self.__mutate(pop)
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


