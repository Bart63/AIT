from typing import List, Tuple


class Particle:
    def __init__(
        self, 
        pos:List[float], 
        inertia_weight:float, 
        cognitive_coef:float, 
        social_coef:float, 
        equation
    ):
        self.pos = pos
        self.best_pos = pos
        self.inertia_weight = inertia_weight
        self.cognitive_coef = cognitive_coef
        self.social_coef = social_coef
        self.velocity:List[float] = []
        self.adaptation = float('inf')
        self.best_adaptation = self.adaptation
        self.equation = equation

    def calc_adaptation(self):
        self.adaptation = self.equation(self.pos)
        if self.adaptation < self.best_adaptation:
            self.best_adaptation = self.adaptation
            self.best_pos = self.pos

    def update_position(
        self, 
        best_swarm_pos:List[int],
        cognitive_rand:float,
        social_rand:float,
        min_max_x:Tuple[float, float]
    ):
        """
        prędkosć = inercja + komponent poznawczy + komponent społeczny 

        gdzie:
        inercja = waga inercji * aktualna prędkość
        waga inercji w przedziale [0, 1]

        komponent poznawczy = przyśpieszenie poznawcze * (najlepsza pozycja cząstku - aktualna pozycja)
        przyśpieszenie poznawcze = współczynnik poznawczy * losowy poziom komponentu poznawczego
        współczynnik poznawczy w przedziale [0, 2]
        losowy poziom komponentu poznawczego w przedziale [0, 1]

        koponent społeczny = przyśpieszenie społeczne * (najlepsza pozycja w roju - aktualna pozycja)
        przyśpieszenie społeczne = współczynnik społeczny * losowy poziom komponentu społecznego
        losowy poziom komponentu społecznego w przedziale [0, 1]
        """
        inertia = [
            self.inertia_weight * v
            for v in self.velocity 
        ]
        
        
        cognitive_acc = self.cognitive_coef * cognitive_rand
        cognitive_comp = [
            cognitive_acc * (bp - p)
            for bp, p in zip(self.best_pos, self.pos)
        ]

        social_acc = self.social_coef * social_rand
        social_comp = [
            social_acc * (bsp - p)
            for bsp, p in zip(best_swarm_pos, self.pos)
        ]

        self.velocity = [
            (inertia[i] if i < len(inertia) else 0) + cc + sc
            for i, (cc, sc) in enumerate(zip(cognitive_comp, social_comp))
        ]
        
        self.pos = [
            max(
                min(
                    p + v,
                    min_max_x[1]
                ), 
                min_max_x[0]
            )
            for p, v in zip(self.pos, self.velocity)
        ]
