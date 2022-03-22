from typing import Dict, List
from particle import Particle
from random import uniform, random

class Swarm:
    def __init__(self, params:Dict):
        self.params = params
        self.best_particle:Particle = None
        self.best_adaptation:float = float('inf')
        self.configurate_particles()

    def configurate_particles(self):
        self.swarm:List[Particle] = [
            Particle(
                (
                    [
                        uniform(
                            self.params['cost_function']['min_x'], 
                            self.params['cost_function']['max_x']
                        )
                        for _ in range(self.params['dimensions'])
                    ]
                ),
                self.params['algorithm']['inertia_weight'],
                self.params['algorithm']['cognitive_coef'],
                self.params['algorithm']['social_coef'],
                list(self.params['cost_function']['equation'].values())[0]
            )
            for _ in range(self.params['population_size'])
        ]
    
    def update_best_particle(self):
        for p in self.swarm:
            p.calc_adaptation()
            best_swarm_adaptation = (
                float('inf') 
                if self.best_particle == None 
                else self.best_particle.best_adaptation
            )
            if p.best_adaptation < best_swarm_adaptation:
                self.best_particle = p 
                self.best_adaptation = p.best_adaptation

    def update_swarm_pos(self):
        cognitive_rand, social_rand = random(), random()
        for p in self.swarm:
            p.update_position(
                self.best_particle.pos,
                cognitive_rand,
                social_rand,
                (
                    self.params['cost_function']['min_x'], 
                    self.params['cost_function']['max_x']
                )
            )

    def get_positions(self):
        return [
            p.pos for p in self.swarm
        ]
