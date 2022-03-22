import random

# Differential Evolution Algorithm
# 1. Mutation.
# 2. Crossover.
# 3. Selection.

class DE:
    def __init__(self, pop, d, sf, cp, m, x_range, cf) -> None:
        self.population_size = pop # N
        self.dims = d # D
        self.scaling_factor = sf # F in [0, 1] for mutation
        self.crossover_prob = cp # Pcr in [0, 1]
        self.method = m # in ['rand' 'best']
        self.x_range = x_range
        self.cost_fun = cf
        self.cost = []
        self.best_cost = 0 # min(y(x))
        self.best_x = 0 # argmin(y(x))

    def vector_init(self, vectors_to_create=0, dims=0, x_range=0):
        if not vectors_to_create:
            vectors_to_create = self.population_size
        if not dims:
            dims = self.dims
        if not x_range:
            x_range = self.x_range

        vectors = []
        distance_range = x_range[1] - x_range[0]
        for i in range(vectors_to_create):
            vectors.append([])
            for _ in range(dims):
                rand_val =  random.random() * distance_range + x_range[0]
                vectors[i].append(rand_val)
        return vectors

    def max_min_correct(self, x):
        for i, xi in enumerate(x):
            x[i] = min(max(xi, self.x_range[0]), self.x_range[1])
    
    def calc_cost(self, input_x):
        self.cost = list(map(self.cost_fun, input_x))
        self.best_cost = min(self.cost)
        self.best_x = input_x[self.cost.index(self.best_cost)]

    # method in ['rand', 'best']
    def reproduction(self, i, input_x):
        # reproduction
        sequence = [val for val in range(len(input_x)) if val != i]
        r1, r2, r3 = random.sample(sequence, 3)

        if self.method == 'best':
            r1 = self.cost.index(self.best_cost)
        
        return r1, r2, r3

    def mutate(self, i, input_x):
        r1, r2, r3 = self.reproduction(i, input_x)
        # mutation
        donor_vector = [] # v vector
        for x1, x2, x3 in zip(input_x[r1], input_x[r2], input_x[r3]):
            donor_vector.append(x1 + self.scaling_factor * (x2 - x3))
        
        self.max_min_correct(donor_vector)
        return donor_vector


    def crossover(self, x, v):
        i_rand = random.randrange(0, self.dims)
        trial_vector = [] # u vector
        for i, (xi, vi) in enumerate(zip(x, v)):
            r = random.random()
            if r <= self.crossover_prob or i == i_rand:
                trial_vector.append(vi)
            else:
                trial_vector.append(xi)
        return trial_vector


    def selection(self, x, u):
        x_val = self.cost_fun(x)
        u_val = self.cost_fun(u)
        if u_val < x_val:
            return u
        return x


def main():
    DE_algo = DE()
    print('Start program!\n')
    input_x = DE_algo.vector_init()
    cost_fun = DE_algo.cost_fun

    value_list = []
    for x in input_x:
        value_list.append(cost_fun(x))
    print(f'Start max is: {max(value_list)}')
    print(f'Start min is: {min(value_list)}\n')
    
    for iter_num in range(DE_algo.iterations):
        print(f'Iteration: {iter_num+1}')

        old_input_x = list(map(list, input_x))
        value_list = []
        for i, x in enumerate(input_x):
            v = DE_algo.mutate(i, old_input_x)
            u = DE_algo.crossover(x, v)  # trial vector
            input_x[i] = DE_algo.selection(x, u)
            value_list.append(cost_fun(input_x[i]))
        print(f'Max is: {max(value_list)}')
        print(f'Min is: {min(value_list)}\n')


if __name__ == '__main__':
    main()
