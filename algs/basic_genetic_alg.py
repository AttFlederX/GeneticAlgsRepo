import numpy as np

class BasicGeneticAlgorithm:
    '''
        Basic implementation of a genetic algorithm for finding the local minimum of a function

        Problem parameters:
            a, b: double             - interval on which to find the local minimum,
            f: (double) -> double    - function to be minimized,
            prec: uint               - precision in positive powers of 10,

            n_p - size of population,
            n_t - size of tournament,
            p_m - probability of mutation,
            p_c - probability of crossover,
            t_max_i - maximum number of unchanging generations
            t_max - maximum number of total generations
    '''
    def __init__(self,
                a,
                b,
                f,
                prec,

                n_p = 100,
                n_t = 2,
                p_m = 0.01,
                p_c = 0.8,
                t_max_i = 10,
                t_max = 1000):

        assert a < b, 'Invalid interval'
        assert prec > 1, 'Invalid precision'

        assert n_t >= 2 and n_p >= n_t, 'Invalid population params' 
        assert 0 <= p_m <= 1 and 0 <= p_c <= 1, 'Invalid probability params' 
        assert t_max_i > 2 and t_max >= t_max_i, 'Invalid stopping conditions' 

        self._a = a
        self._b = b

        self._f = f
        self._prec = prec

        self._n_p = n_p 
        self._n_t = n_t 
        self._p_m = p_m 
        self._p_c = p_c 
        self._t_max_i = t_max_i 
        self._t_max = t_max

        # size of an individual
        self._n = 0


    ## Algorithm functions

    def __find_size_of_individual(self):
        '''
            Finds the length of the bit word required for the given precision, 
            by bitshifting the overall number of points on an interval
        '''
        numOfPoints = int((self._b - self._a) * (10 ** self._prec))
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
            # generate a zero-padded bitword of size n
            newIndiv = f"{np.random.randint(2 ** self._n):b}"
            newIndiv = newIndiv.zfill(self._n)

            pop.append(newIndiv)

        return pop


    def __binary_to_digit(self, v):
        '''
            Converts the bitword into a position within the interval
        '''
        vInt = int(v, 2) # convert the bitword into a decimal integer
        return self._a + ( ((self._b - self._a) / (2**self._n - 1)) * vInt )

    def __fitness(self, v):
        '''
            Fitness function for an individual (less => more fit)
        '''
        return self._f(self.__binary_to_digit(v))

    def __population_fitness(self, pop):
        '''
            For a given population, returns its best fitness & the individual which has it
        '''
        # find the smallest (i.e. best) fitness in a population
        min = np.min(list(map(lambda v: self.__fitness(v), pop)))
        # find the individial with the smallest fitness
        minIndiv = next(filter(lambda v: self.__fitness(v) <= min, pop))

        return min, self.__binary_to_digit(minIndiv)


    ## Generational operators

    def __select(self, pop):
        '''
            Perform a population selection using the tournament method

                - Select n_t individuals from the population
                - Add the most fit individual to the new population
                - Repeat until the new population's size is n_p

                (individuals are not required to be unique)
        '''
        selectedPop = list()
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

                randBitIdx = np.random.randint(self._n)

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
            Performs crossovers between random pairs of individuals

                - Pick two random individuals from the population
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

        while len(crossoverPop) < self._n_p:
            # pick two random individuals
            v_i = pop[np.random.randint(self._n_p)]
            v_j = pop[np.random.randint(self._n_p)]

            if np.random.random() <= self._p_c:
                # pick a split point
                splitPoint = np.random.randint(self._n)

                # split the individuals
                v_i_firstPart = v_i[:splitPoint]
                v_i_secondPart = v_i[splitPoint:]

                v_j_firstPart = v_j[:splitPoint]
                v_j_secondPart = v_j[splitPoint:]

                # put the merged ones into the new population
                crossoverPop.append(f"{v_i_firstPart}{v_j_secondPart}")
                crossoverPop.append(f"{v_j_firstPart}{v_i_secondPart}")
            else:
                # add without change
                crossoverPop.append(v_i)
                crossoverPop.append(v_j)

        return crossoverPop

        
    ## Main entry point

    def solve(self, verbose_mode=False):
        # find bitword length
        self._n = self.__find_size_of_individual()

        # generate initial population
        pop = self.__generate_population()

        # initialize iteration counter
        t = 0
        # unchanging iteration counter
        t_i = 0
        # fitness stack
        popFitnessStack = []

        # begin iterating
        while t < self._t_max:
            # perform generational changes
            pop = self.__select(pop)
            pop = self.__mutate(pop)
            pop = self.__crossover(pop)

            # determine best fitness in a population & add it to the stack
            popFitnessVal, bestIndiv = self.__population_fitness(pop)
            popFitnessStack.append(popFitnessVal)

            if verbose_mode == True:
                print(f'Iteration #{t}: best fitness {popFitnessVal} at x = {bestIndiv}')

            if t > 1:
                # if the best fitness difference is small, increment the unchanged iteration counter,
                # else, reset it
                if abs(popFitnessStack[t] - popFitnessStack[t - 1] < 10 ** -self._prec):
                    t_i += 1
                else:
                    t_i = 0

            # if max unchanged iterations reached, stop
            if t_i >= self._t_max_i:
                break

            t += 1

        # return best fitness of final population
        res, x = self.__population_fitness(pop)
        return res, x, t


