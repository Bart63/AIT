import random

# Differential Evolution Algorithm
# 1. Mutation.
# 2. Crossover.
# 3. Selection.

population_size = 10 # N
dims = 2 # D
scaling_factor = 0.1 # F in [0, 2] for mutation
crossover_prob = 0.2 # Pcr in [0, 1]
iterations = 10

x_range = [-100, 100]


def vector_init(vectors_to_create, dims, x_range):
    vectors = []
    distance_range = x_range[1] - x_range[0]
    for i in range(vectors_to_create):
        vectors.append([])
        for _ in range(dims):
            rand_val =  random.random() * distance_range + x_range[0]
            vectors[i].append(rand_val)
    return vectors


def sphere_fun(x):
    def sphere_mapper(x):
        return x*x
    return sum(map(sphere_mapper, x))


def max_min_correct(x):
    for i, xi in enumerate(x):
        x[i] = min(max(xi, x_range[0]), x_range[1])


def mutate(i, input_x):
    sequence = [val for val in range(len(input_x)) if val != i]
    r1, r2, r3 = random.sample(sequence, 3)

    donor_vector = [] # v vector
    for x1, x2, x3 in zip(input_x[r1], input_x[r2], input_x[r3]):
        donor_vector.append(x1 + scaling_factor * (x2 - x3))
    
    max_min_correct(donor_vector)
    return donor_vector


def crossover(x, v):
    i_rand = random.randrange(0, dims)
    trial_vector = [] # u vector
    for i, (xi, vi) in enumerate(zip(x, v)):
        r = random.random()
        if r <= crossover_prob or i == i_rand:
            trial_vector.append(vi)
        else:
            trial_vector.append(xi)
    return trial_vector


def selection(x, u, cost_fun):
    x_val = cost_fun(x)
    u_val = cost_fun(u)
    if u_val < x_val:
        return u
    return x


def main():
    print('Start program!\n')
    input_x = vector_init(population_size, dims, x_range)
    cost_fun = sphere_fun

    value_list = []
    for x in input_x:
        value_list.append(cost_fun(x))
    print(f'Start max is: {max(value_list)}')
    print(f'Start min is: {min(value_list)}\n')
    
    for iter_num in range(iterations):
        print(f'Iteration: {iter_num+1}')

        old_input_x = list(map(list, input_x))
        value_list = []
        for i, x in enumerate(input_x):
            v = mutate(i, old_input_x)
            u = crossover(x, v)
            input_x[i] = selection(x, u, cost_fun)
            value_list.append(cost_fun(input_x[i]))
        print(f'Max is: {max(value_list)}')
        print(f'Min is: {min(value_list)}\n')


if __name__ == '__main__':
    main()
