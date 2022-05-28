import random
import math


class GLPSO_OBL:
    def __init__(self, params) -> None:
        self.D = params['dimensions']
        self.M = params['population_size']
        self.x_min = params['cost_function']['min_x']
        self.x_max = params['cost_function']['max_x']
        self.pm = params['algorithm']['mutation_prob']
        self.w = params['algorithm']['inertia_weight']
        self.c = params['algorithm']['c']
        self.c1 = params['algorithm']['c1']
        self.c2 = params['algorithm']['c2']
        self.sg = params['algorithm']['sg']
        self.eval = list(params['cost_function']['equation'].values())[0]
        self.generate_vars()

    def generate_vars(self):
        self.X = [[random.uniform(self.x_min, self.x_max) for _ in range(self.D)] for _ in range(self.M)]
        self.V = [[random.uniform(-1, 1) for _ in range(self.D)] for _ in range(self.M)]
        self.P = [x_arr.copy() for x_arr in self.X] # personal best
        eval_values = [self.eval(p) for p in self.P]
        self.best_eval = min(eval_values)
        self.G = self.X[eval_values.index(min(eval_values))].copy() # global best
        self.G_eval = self.eval(self.G)
        self.SG = [0 for _ in range(self.M)]

    def do_epoch(self):
        for idx, x_arr in enumerate(self.X):
            # Crossover
            O = []
            o_d = 0
            for d in range(self.D):
                chosen_particle = random.randint(0, self.M-1)
                if self.eval(x_arr) < self.eval(self.X[chosen_particle]):
                    r_d = random.random()
                    o_d = r_d * self.P[idx][d] + (1-r_d) * self.G[d]
                else:
                    o_d = self.P[chosen_particle][d]
                O.append(o_d)
            
            # Mutation
            for d in range(self.D):
                if random.random() < self.pm:
                    O[d] = self.x_min + self.x_max - O[d]

            # E vector
            E = []
            for d in range(self.D):
                r1, r2 = random.random(), random.random()
                numerator = self.c1 * r1 * self.P[idx][d] + self.c2 * r2 * self.G[d]
                denumerator = self.c1*r1 + self.c2*r2
                E.append(numerator/denumerator)

            # Selection
            E_eval = self.eval(E)
            if self.eval(O) < E_eval:
                E = O
            
            if E_eval <= self.eval(self.P[idx]):
                self.SG[idx] += 1
                if self.SG[idx] == self.sg:
                    self.SG[idx] = 0
                    tournament_indexes = random.sample([num for num in range(self.M)], int(math.ceil(0.2 * self.M)))
                    tournament_evals = [self.eval(self.X[t_idx]) for t_idx in tournament_indexes]
                    winner_tournament_idx = tournament_evals.index(min(tournament_evals))
                    winner_idx = tournament_indexes[winner_tournament_idx]
                    winner = self.X[winner_idx]
                    E = winner
            else:
                self.SG[idx] = 0
            
            # Particle update
            for d in range(self.D):
                r_d = random.random()
                self.V[idx][d] = self.w * self.V[idx][d] + self.c * r_d * (E[d] - self.X[idx][d])
                self.X[idx][d] += self.V[idx][d]
            
            x_eval = self.eval(self.X[idx])
            if x_eval < self.eval(self.P[idx]):
                self.P[idx] = self.X[idx].copy()
                if x_eval < self.G_eval:
                    self.best_eval = x_eval
                    self.G = self.X[idx].copy()
                    self.G_eval = x_eval


def main():
    params = {
        'population_size' : 10,
        'dimensions' : 5,
        'cost_function' : {
            'min_x' : -5,
            'max_x' : 5,
            'equation' : {
                "Sphere" : 
                    lambda x_args: sum(map(
                        lambda x: x*x, 
                        x_args
                    )),
            }
        },
        "algorithm": {
            "mutation_prob" : 0.9, # w przedziale [0, 1]
            "inertia_weight" : 1.9, # w przedziale [0, 2]
            "amplification_factor": 0.5, # F w przedziale [0, 1]
            "c" : 0.1, # w przedziale [0, 2]
            "c1" : 0.1, # w przedziale [0, 2]
            "c2" : 0.1, # w przedziale [0, 2]
            "sg" : 7
        }
    }
    glpso_obl = GLPSO_OBL(params)
    print(glpso_obl.eval(glpso_obl.G))
    glpso_obl.do_epoch()
    print(glpso_obl.eval(glpso_obl.G))


if __name__ == '__main__':
    main()
