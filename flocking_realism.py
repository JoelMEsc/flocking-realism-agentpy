"""
Aumentando el realismo en modelos de AgentPy: Flocking Behavior.

Código del modelo, sin el análisis. Es el mismo código de `flocking_realism.ipynb`
(que contiene la explicación, las justificaciones y las figuras), organizado como módulo:

1. Modelo base de la AgentPy Model Library (sección 1.0 del notebook).
2. Modelo extendido por etapas: referencia con escala realista (3.0), percepción (3.1),
   roles (3.2), energía (3.3) y modelo completo (3.4).
3. Métricas y KPIs (secciones 3.0 y 5).
4. Animación con slider del modelo (3.4).
5. Calibración (4.1) e infraestructura y configuraciones de los experimentos (6).
6. Pruebas de verificación (7.1).

Uso desde la línea de comandos:
    python flocking_realism.py                             # una corrida del modelo completo y sus KPIs
    python flocking_realism.py --animacion                 # animación con slider en animacion.html
    python flocking_realism.py --pruebas                   # pruebas de verificación del código
    python flocking_realism.py --experimentos              # carga los experimentos E1 a E6 y muestra un resumen
    python flocking_realism.py --experimentos --recalcular # vuelve a correr los experimentos (~30 min)

Uso como módulo:
    from flocking_realism import ModeloExtendido, parametros_extendido
    salida = ModeloExtendido(parametros_extendido).run()
"""

import argparse
import os

import agentpy as ap
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# ============================================================================
# 1. Modelo base de la AgentPy Model Library (sección 1.0)
# ============================================================================

def normalize(v):
    """ Normalize a vector to length 1. """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

class Boid(ap.Agent):
    """ An agent with a position and velocity in a continuous space,
    who follows Craig Reynolds three rules of flocking behavior;
    plus a fourth rule to avoid the edges of the simulation space. """

    def setup(self):

        self.velocity = normalize(
            self.model.nprandom.random(self.p.ndim) - 0.5)

    def setup_pos(self, space):

        self.space = space
        self.neighbors = space.neighbors
        self.pos = space.positions[self]

    def update_velocity(self):

        pos = self.pos
        ndim = self.p.ndim

        # Rule 1 - Cohesion
        nbs = self.neighbors(self, distance=self.p.outer_radius)
        nbs_len = len(nbs)
        nbs_pos_array = np.array(nbs.pos)
        nbs_vec_array = np.array(nbs.velocity)
        if nbs_len > 0:
            center = np.sum(nbs_pos_array, 0) / nbs_len
            v1 = (center - pos) * self.p.cohesion_strength
        else:
            v1 = np.zeros(ndim)

        # Rule 2 - Seperation
        v2 = np.zeros(ndim)
        for nb in self.neighbors(self, distance=self.p.inner_radius):
            v2 -= nb.pos - pos
        v2 *= self.p.seperation_strength

        # Rule 3 - Alignment
        if nbs_len > 0:
            average_v = np.sum(nbs_vec_array, 0) / nbs_len
            v3 = (average_v - self.velocity) * self.p.alignment_strength
        else:
            v3 = np.zeros(ndim)

        # Rule 4 - Borders
        v4 = np.zeros(ndim)
        d = self.p.border_distance
        s = self.p.border_strength
        for i in range(ndim):
            if pos[i] < d:
                v4[i] += s
            elif pos[i] > self.space.shape[i] - d:
                v4[i] -= s

        # Update velocity
        self.velocity += v1 + v2 + v3 + v4
        self.velocity = normalize(self.velocity)

    def update_position(self):

        self.space.move_by(self, self.velocity)

class BoidsModel(ap.Model):
    """
    An agent-based model of animals' flocking behavior,
    based on Craig Reynolds' Boids Model [1]
    and Conrad Parkers' Boids Pseudocode [2].

    [1] http://www.red3d.com/cwr/boids/
    [2] http://www.vergenet.net/~conrad/boids/pseudocode.html
    """

    def setup(self):
        """ Initializes the agents and network of the model. """

        self.space = ap.Space(self, shape=[self.p.size]*self.p.ndim)
        self.agents = ap.AgentList(self, self.p.population, Boid)
        self.space.add_agents(self.agents, random=True)
        self.agents.setup_pos(self.space)

    def step(self):
        """ Defines the models' events per simulation step. """

        self.agents.update_velocity()   # Adjust direction
        self.agents.update_position()   # Move into new direction

parameters2D = {
    'size': 50,
    'seed': 123,
    'steps': 200,
    'ndim': 2,
    'population': 200,
    'inner_radius': 3,
    'outer_radius': 10,
    'border_distance': 10,
    'cohesion_strength': 0.005,
    'seperation_strength': 0.1,
    'alignment_strength': 0.3,
    'border_strength': 0.5
}

# ============================================================================
# 2. Etapa 0: modelo de referencia con escala realista (sección 3.0)
# ============================================================================

G = 9.81  # Aceleración de la gravedad (m/s²)


def desplazamiento_toroidal(origen, destino, size):
    """ Vector más corto de `origen` a `destino` en un torus cuadrado de lado `size`.
    Funciona con puntos sueltos o con arreglos de puntos (broadcasting de numpy). """
    d = np.asarray(destino) - np.asarray(origen)
    return d - size * np.round(d / size)


def rotar_hacia(heading, deseada, max_angulo):
    """ Gira el vector unitario `heading` hacia la dirección `deseada` (2D),
    como máximo `max_angulo` radianes. Devuelve el nuevo rumbo y el ángulo girado. """
    angulo = np.arctan2(heading[0] * deseada[1] - heading[1] * deseada[0],
                        heading @ deseada)
    giro = np.clip(angulo, -max_angulo, max_angulo)
    c, s = np.cos(giro), np.sin(giro)
    nuevo = np.array([c * heading[0] - s * heading[1],
                      s * heading[0] + c * heading[1]])
    return nuevo, giro

class BoidReferencia(ap.Agent):
    """ Boid con escala física: 1 unidad = 1 m y 1 paso = dt segundos.

    Diferencias con el modelo base:
    - Dirección (`heading`) y rapidez (`speed`) separadas; `velocity` = heading × speed.
    - Giro máximo por paso, limitado por el factor de carga máximo del vuelo.
    - Vecindad en un torus: las reglas usan el desplazamiento más corto.
    - Sin regla de bordes, porque el torus no tiene paredes. """

    def setup(self):
        sd = self.p.get('init_heading_sd', None)
        if sd is None:
            # Parvada sin formar: rumbo aleatorio, como en el modelo base
            self.heading = normalize(self.model.nprandom.random(self.p.ndim) - 0.5)
        else:
            # Parvada ya formada: rumbo común con dispersión normal de `sd` grados
            angulo = self.model.direccion_inicial + self.model.nprandom.normal(0, np.radians(sd))
            self.heading = np.array([np.cos(angulo), np.sin(angulo)])
        self.speed = self.p.speed_pref
        self.velocity = self.heading * self.speed
        self.turn = 0.0  # Ángulo girado en el último paso (rad)

    def setup_pos(self, space):
        self.space = space
        self.pos = space.positions[self]

    def max_turn(self):
        """ Giro máximo por paso (rad) a la rapidez actual.
        Aceleración lateral máxima a = g·√(n² − 1); giro Δθ = a·dt / v,
        con v = speed / dt la rapidez en m/s. """
        a_max = G * np.sqrt(self.p.max_load_factor ** 2 - 1)
        return a_max * self.p.dt ** 2 / self.speed

    def vecinos(self):
        """ Vecinos que percibe el ave: desplazamientos hacia ellos y sus velocidades.
        En la referencia, todas las aves a menos de outer_radius. """
        nbs = self.space.neighbors(self, distance=self.p.outer_radius)
        if len(nbs) == 0:
            return np.zeros((0, self.p.ndim)), np.zeros((0, self.p.ndim))
        disp = desplazamiento_toroidal(self.pos, np.array(nbs.pos), self.p.size)
        return disp, np.array(nbs.velocity)

    def social_force(self):
        """ Suma de las reglas de cohesión, separación y alineación. """
        disp, vel = self.vecinos()
        force = np.zeros(self.p.ndim)
        if len(disp) == 0:
            return force
        dist = np.linalg.norm(disp, axis=1)

        # Regla 1 - Cohesión: hacia el centro de los vecinos
        force += disp.mean(axis=0) * self.p.cohesion_strength
        # Regla 2 - Separación: alejarse de los vecinos dentro de inner_radius
        force -= disp[dist < self.p.inner_radius].sum(axis=0) * self.p.separation_strength
        # Regla 3 - Alineación: igualar la velocidad promedio de los vecinos
        force += (vel.mean(axis=0) - self.velocity) * self.p.alignment_strength
        return force

    def update_velocity(self):
        deseada = self.velocity + self.social_force()
        self.heading, self.turn = rotar_hacia(self.heading, deseada, self.max_turn())
        self.velocity = self.heading * self.speed

    def update_position(self):
        self.space.move_to(self, np.mod(self.pos + self.velocity, self.p.size))

class ModeloReferencia(ap.Model):
    """ Modelo de boids con escala realista en un torus. Es la referencia
    contra la que se comparan las mejoras. """

    agent_class = BoidReferencia

    def setup(self):
        assert self.p.ndim == 2, 'El modelo de referencia es bidimensional'
        self.direccion_inicial = self.nprandom.uniform(0, 2 * np.pi)  # Rumbo común de la parvada (rad)
        self.space = ap.Space(self, shape=[self.p.size] * 2, torus=True)
        self.agents = ap.AgentList(self, self.p.population, self.agent_class)
        self.space.add_agents(self.agents, positions=self.posiciones_iniciales())
        self.agents.setup_pos(self.space)

    def posiciones_iniciales(self):
        """ Parche cuadrado en el centro del espacio. Para puntos uniformes con
        densidad ρ, la distancia media al vecino más cercano es ≈ 0.5/√ρ; el lado
        del parche se elige para que esa distancia sea `init_nnd`. """
        lado = 2 * self.p.init_nnd * np.sqrt(self.p.population)
        centro = self.p.size / 2
        return [centro + (self.nprandom.random(2) - 0.5) * lado
                for _ in range(self.p.population)]

    def step(self):
        self.agents.update_velocity()
        self.agents.update_position()

def distancias_toroidales(m):
    """ Matriz de distancias entre todos los agentes, por el camino más corto en el torus. """
    pos = np.array(list(m.space.positions.values()))
    return np.linalg.norm(desplazamiento_toroidal(pos[:, None], pos[None], m.p.size), axis=-1)


def metricas_parvada(m, d=None):
    """ Indicadores del estado de la parvada en el paso actual.
    `d` es opcional: la matriz de distancias ya calculada, con infinito en la diagonal. """
    if d is None:
        d = distancias_toroidales(m)
        np.fill_diagonal(d, np.inf)
    nnd = d.min(axis=1)
    n_grupos, etiquetas = connected_components(csr_matrix(d < m.p.outer_radius), directed=False)
    tamanos = np.bincount(etiquetas)
    mayor = etiquetas == tamanos.argmax()
    d_mayor = d[np.ix_(mayor, mayor)]
    extension = d_mayor[np.isfinite(d_mayor)].max() if mayor.sum() > 1 else 0.0
    return {
        'polarizacion': np.linalg.norm(np.mean([a.heading for a in m.agents], axis=0)),
        'dist_vecino': nnd.mean(),
        'frac_colision': (nnd < m.p.collision_distance).mean(),
        'n_grupos': n_grupos,
        'frac_grupo_mayor': tamanos.max() / len(etiquetas),
        'extension': extension,
    }


class ModeloReferenciaConDatos(ModeloReferencia):
    """ Modelo de referencia que registra las métricas en cada paso. """

    def update(self):
        for k, v in metricas_parvada(self).items():
            self.record(k, v)

parametros_ref = {
    # Simulación
    'seed': 123,
    'steps': 1000,             # 1000 pasos = 100 s de vuelo
    'ndim': 2,
    'population': 200,
    # Escala
    'dt': 0.1,                 # s por paso
    'size': 200,               # m, lado del torus
    'init_nnd': 1.0,           # m, distancia al vecino en la posición inicial
    'init_heading_sd': 16.4,   # grados, dispersión inicial de rumbos (None = rumbos aleatorios)
    'collision_distance': 0.38,  # m, distancia mínima observada entre aves
    # Vuelo
    'speed_pref': 1.0,         # m/paso (= 10 m/s)
    'max_load_factor': 2.0,    # g
    # Reglas
    'inner_radius': 1.0,         # m, calibrado en la sección 4.1
    'outer_radius': 10,        # m
    'cohesion_strength': 0.005,
    'separation_strength': 0.4,  # calibrado en la sección 4.1
    'alignment_strength': 0.3,
}

# ============================================================================
# 3. Etapa 1: percepción realista (sección 3.1)
# ============================================================================

class BoidPercepcion(BoidReferencia):
    """ Boid con percepción realista:
    - ve solo dentro de su campo visual (`vision_angle`, centrado en su rumbo),
      es decir, no percibe a las aves en su punto ciego trasero;
    - ve hasta `vision_radius` metros;
    - responde solo a sus `n_neighbors` vecinos visibles más cercanos
      (interacción topológica). Con `n_neighbors = None` responde a todos.

    Con vision_angle = 360, n_neighbors = None y vision_radius = outer_radius
    se comporta exactamente igual que BoidReferencia. """

    def vecinos(self):
        p = self.p
        radio = p.get('vision_radius', p.outer_radius)
        angulo = p.get('vision_angle', 360)
        k = p.get('n_neighbors', None)
        if k is None and angulo >= 360 and radio == p.outer_radius:
            return super().vecinos()

        tree = self.space.kdtree
        agentes = self.space._sorted_agents  # Agentes en el orden de los puntos del árbol k-d
        n = len(agentes)
        candidatos = n if k is None else min(n, 3 * k + 1)
        while True:
            if k is None:
                idx = np.array(tree.query_ball_point(self.pos, radio), dtype=int)
            else:
                dist, idx = tree.query(self.pos, k=candidatos, distance_upper_bound=radio)
                idx = np.atleast_1d(idx)[np.isfinite(np.atleast_1d(dist))]
            idx = idx[[agentes[i] is not self for i in idx]]

            disp = desplazamiento_toroidal(self.pos, tree.data[idx], p.size)
            dist = np.linalg.norm(disp, axis=1)
            if angulo < 360:
                # Visible si el ángulo entre el rumbo y la dirección al vecino es ≤ angulo/2
                cos_vecino = disp @ self.heading / np.maximum(dist, 1e-12)
                visible = cos_vecino >= np.cos(np.radians(angulo / 2))
                idx, disp, dist = idx[visible], disp[visible], dist[visible]

            # Si la búsqueda se truncó y quedaron menos de k visibles, se amplía
            if k is None or len(idx) >= k or candidatos >= n:
                break
            candidatos = min(n, 2 * candidatos)

        if k is not None:
            orden = np.argsort(dist)[:k]
            idx, disp = idx[orden], disp[orden]
        if len(idx) == 0:
            return np.zeros((0, p.ndim)), np.zeros((0, p.ndim))
        return disp, np.array([agentes[i].velocity for i in idx])

class ModeloPercepcion(ModeloReferencia):
    """ Modelo de la Etapa 1: referencia + percepción realista. """
    agent_class = BoidPercepcion


class ModeloPercepcionConDatos(ModeloReferenciaConDatos):
    """ Etapa 1 con el registro de métricas de la sección 3.0. """
    agent_class = BoidPercepcion

parametros_percepcion = dict(
    parametros_ref,
    vision_radius=50,   # m
    vision_angle=300,   # grados; punto ciego trasero de 60°
    n_neighbors=7,      # vecinos con los que interactúa cada ave
)

# ============================================================================
# 4. Etapa 2: roles (sección 3.2)
# ============================================================================

ROLES = ('lider', 'adulto', 'juvenil')


def conteo_por_rol(fracciones, n):
    """ Número exacto de aves por rol. Redondea hacia abajo y reparte el resto
    entre los roles con mayor parte fraccionaria, para que la suma sea `n`. """
    ideal = {r: fracciones[r] * n for r in fracciones}
    conteo = {r: int(np.floor(x)) for r, x in ideal.items()}
    faltan = n - sum(conteo.values())
    for r in sorted(ideal, key=lambda r: ideal[r] - conteo[r], reverse=True)[:faltan]:
        conteo[r] += 1
    return conteo

class BoidRoles(BoidPercepcion):
    """ Boid con un rol: 'lider', 'adulto' o 'juvenil'.

    - Cada ave tiene su propia copia de parámetros: los del modelo, sobrescritos
      por los de su rol. Un parámetro con sufijo `__<rol>` (por ejemplo
      `alignment_strength__juvenil`) reemplaza al parámetro base solo para ese
      rol. Así cualquier regla puede tener un valor distinto por rol sin cambiar
      su código, y ese valor se puede barrer en un experimento.
    - Los líderes son adultos informados: además de las reglas sociales, giran
      hacia la dirección preferida del modelo con peso `goal_weight`
      (Couzin et al., 2005). """

    def setup(self):
        self.role = next(self.model.roles_por_asignar)
        sufijo = f'__{self.role}'
        propios = {k[:-len(sufijo)]: v for k, v in self.model.p.items() if k.endswith(sufijo)}
        self.p = ap.AttrDict({**self.model.p, **propios})
        super().setup()

    def goal_force(self):
        """ Atracción hacia la dirección preferida (solo líderes), con la magnitud de la velocidad. """
        if self.role != 'lider' or self.p.goal_weight == 0:
            return np.zeros(self.p.ndim)
        return self.p.goal_weight * self.speed * self.model.goal_dir

    def social_force(self):
        return super().social_force() + self.goal_force()

class ModeloRoles(ModeloPercepcion):
    """ Modelo de la Etapa 2: percepción realista + roles.

    Las proporciones son exactas: `frac_lider` y `frac_juvenil` fijan cuántas aves
    tiene cada rol y el resto son adultos; el orden se baraja con el generador del
    modelo. Si hay líderes, el modelo elige al azar una dirección preferida
    (`goal_dir`), común a todos ellos y fija durante la corrida. """

    agent_class = BoidRoles

    def setup(self):
        fracciones = {'lider': self.p.get('frac_lider', 0.0),
                      'juvenil': self.p.get('frac_juvenil', 0.0)}
        fracciones['adulto'] = 1 - sum(fracciones.values())
        assert fracciones['adulto'] >= 0, 'frac_lider + frac_juvenil no puede superar 1'
        self.conteo_roles = conteo_por_rol(fracciones, self.p.population)
        roles = [r for r in ROLES for _ in range(self.conteo_roles[r])]
        self.random.shuffle(roles)
        self.roles_por_asignar = iter(roles)
        super().setup()
        if self.conteo_roles['lider'] > 0:
            angulo = self.random.uniform(0, 2 * np.pi)
            self.goal_dir = np.array([np.cos(angulo), np.sin(angulo)])


class ModeloRolesConDatos(ModeloRoles):
    """ Etapa 2 con el registro de la sección 3.0, más la precisión de rumbo. """

    def update(self):
        ModeloReferenciaConDatos.update(self)
        if hasattr(self, 'goal_dir'):
            self.record('precision_rumbo', np.mean([a.heading @ self.goal_dir for a in self.agents]))

parametros_roles = dict(
    parametros_percepcion,
    frac_lider=0.10,      # líderes informados (adultos)
    frac_juvenil=0.45,    # juveniles; el resto son adultos
    goal_weight=0.05,     # peso de la dirección preferida (solo líderes)
)

# ============================================================================
# 5. Etapa 3: energía y modelo completo (secciones 3.3 y 3.4)
# ============================================================================

def costo_rapidez(v):
    """ Potencia de vuelo relativa a la de 10 m/s, interpolando las mediciones de
    Ward et al. (2001) en estorninos: 10,4 W a 6,3 m/s y 14,9 W a 14,4 m/s. """
    potencia = 10.4 + (14.9 - 10.4) * (v - 6.3) / (14.4 - 6.3)
    potencia_10 = 10.4 + (14.9 - 10.4) * (10 - 6.3) / (14.4 - 6.3)
    return potencia / potencia_10


def costo_giro(v, giro, dt):
    """ Factor de carga al cuadrado, n² = 1 + (a/g)², con aceleración lateral
    a = v·|giro|/dt. Volar recto cuesta 1; un giro a 2 g cuesta 4 (Usherwood et al., 2011). """
    a = v * abs(giro) / dt
    return 1 + (a / G) ** 2


def costo_racha(k, s_max, tau):
    """ Multiplicador por giros consecutivos: 1 para un giro aislado, crece con cada
    giro seguido y se satura en `s_max`. `k` es el número de giros seguidos. """
    return 1 + (s_max - 1) * (1 - np.exp(-max(k - 1, 0) / tau))

class BoidEnergia(BoidRoles):
    """ Boid con energía (0 a 1) y rapidez variable.

    - La rapidez deseada sale de las reglas sociales (que incluyen igualar la
      velocidad de los vecinos) más una tendencia hacia `speed_pref`, y se recorta
      entre `v_min` y la rapidez máxima que permite la energía.
    - Cada paso gasta energía según la rapidez, los giros y los giros seguidos, por
      `cost_factor` (mayor en juveniles), y recupera `recovery_rate`, propio de cada ave.
    - Con energía menor que `e_crit`, la rapidez máxima baja linealmente hasta `v_min`.

    Con fatigue_scale = 0 y v_min = v_max = speed_pref se comporta igual que BoidRoles. """

    def setup(self):
        super().setup()
        self.energy = 1.0
        self.racha = 0  # Giros seguidos mayores que turn_threshold

    def v_max_actual(self):
        """ Rapidez máxima según la energía (m/paso). """
        v_min, v_max = self.p.get('v_min', self.p.speed_pref), self.p.get('v_max', self.p.speed_pref)
        fraccion = min(1.0, self.energy / self.p.e_crit) if self.p.get('e_crit', 0) > 0 else 1.0
        return v_min + (v_max - v_min) * fraccion

    def update_velocity(self):
        deseada = self.velocity + self.social_force()
        self.heading, self.turn = rotar_hacia(self.heading, deseada, self.max_turn())

        # Rapidez: la que piden las reglas, más una tendencia hacia la preferida
        objetivo = np.linalg.norm(deseada) + self.p.get('speed_relax', 0) * (self.p.speed_pref - self.speed)
        self.speed = float(np.clip(objetivo, self.p.get('v_min', self.p.speed_pref), self.v_max_actual()))
        self.velocity = self.heading * self.speed
        self.gastar_energia()

    def gastar_energia(self):
        escala = self.p.get('fatigue_scale', 0)
        if escala == 0:
            return
        self.racha = self.racha + 1 if abs(np.degrees(self.turn)) > self.p.turn_threshold else 0
        v = self.speed / self.p.dt  # m/s
        self.costo = (self.p.cost_factor * costo_rapidez(v) * costo_giro(v, self.turn, self.p.dt)
                      * costo_racha(self.racha, self.p.streak_max, self.p.streak_tau))
        self.energy = float(np.clip(self.energy + escala * (self.recovery_rate - self.costo), 0, 1))

class ModeloEnergia(ModeloRoles):
    """ Modelo de la Etapa 3: roles + energía. La tasa de recuperación de cada ave
    sigue una distribución normal truncada en 0, con media `recovery_mean` y
    coeficiente de variación `recovery_cv`. """

    agent_class = BoidEnergia

    def setup(self):
        super().setup()
        media, cv = self.p.get('recovery_mean', 1.0), self.p.get('recovery_cv', 0.0)
        for a in self.agents:
            a.recovery_rate = max(0.0, self.nprandom.normal(media, cv * media)) if cv > 0 else media


class ModeloEnergiaConDatos(ModeloEnergia):
    """ Etapa 3 con el registro de la sección 3.0, más energía y rapidez por rol.
    Al final registra el rol, la energía y la distancia recorrida de cada ave. """

    def update(self):
        ModeloReferenciaConDatos.update(self)
        for rol in ROLES:
            aves = [a for a in self.agents if a.role == rol]
            if aves:
                self.record(f'energia_{rol}', np.mean([a.energy for a in aves]))
                self.record(f'rapidez_{rol}', np.mean([a.speed for a in aves]) / self.p.dt)
        for a in self.agents:
            a.distancia = getattr(a, 'distancia', 0.0) + (a.speed if self.t > 0 else 0.0)

    def end(self):
        self.agents.record(['role', 'energy', 'distancia'])

parametros_energia = dict(
    parametros_roles,
    # Rapidez
    v_min=0.69,               # m/paso (6,9 m/s)
    v_max=1.5,                # m/paso (15 m/s)
    speed_relax=0.1,          # tendencia a volver a speed_pref
    # Gasto
    fatigue_scale=0.008,      # compresión del tiempo de fatiga
    cost_factor=1.0,
    cost_factor__juvenil=1.14,
    turn_threshold=5,         # grados por paso para contar en la racha
    streak_max=2.0,           # multiplicador máximo por giros seguidos
    streak_tau=5,             # giros seguidos para acercarse al máximo
    # Recuperación
    recovery_mean=1.26,       # calibrada en la sección 3.3
    recovery_cv=0.10,
    e_crit=0.3,               # energía bajo la cual baja la rapidez máxima
)

ModeloExtendido = ModeloEnergia
parametros_extendido = dict(parametros_energia)

# ============================================================================
# Animación del modelo (sección 3.4)
# ============================================================================

COLORES_ROL = {'lider': 'tab:red', 'adulto': 'tab:blue', 'juvenil': 'tab:orange'}


def centro_parvada(m):
    """ Centro del grupo mayor en el torus: promedio de los desplazamientos desde una de sus aves. """
    d = distancias_toroidales(m)
    _, etiquetas = connected_components(csr_matrix(d < m.p.outer_radius), directed=False)
    pos = np.array(list(m.space.positions.values()))
    grupo = pos[etiquetas == np.bincount(etiquetas).argmax()]
    return grupo[0] + desplazamiento_toroidal(grupo[0], grupo, m.p.size).mean(axis=0)


def cuadro_parvada(m, ax, ventana=60):
    """ Dibuja un paso de la simulación con una cámara que sigue al grupo mayor.
    Color: rol del ave. Opacidad: energía (más transparente = más cansada). """
    ax.clear()
    centro = centro_parvada(m)
    for rol, color in COLORES_ROL.items():
        aves = [a for a in m.agents if getattr(a, 'role', 'adulto') == rol]
        if not aves:
            continue
        rel = desplazamiento_toroidal(centro, np.array([a.pos for a in aves]), m.p.size)
        rumbo = np.array([a.heading for a in aves])
        energia = np.array([getattr(a, 'energy', 1.0) for a in aves])
        ax.quiver(rel[:, 0], rel[:, 1], rumbo[:, 0], rumbo[:, 1], color=color, alpha=0.25 + 0.75 * energia,
                  angles='xy', scale=45, width=0.005, label=rol)
    kpi = metricas_parvada(m)
    texto = f"t = {m.t * m.p.dt:.1f} s   polarización = {kpi['polarizacion']:.2f}   grupos = {kpi['n_grupos']}"
    if hasattr(m.agents[0], 'energy'):
        texto += f"   energía media = {np.mean([a.energy for a in m.agents]):.2f}"
    ax.set(xlim=(-ventana, ventana), ylim=(-ventana, ventana), aspect='equal',
           xlabel='m (relativo al centro del grupo mayor)', title=texto)
    ax.legend(loc='upper right', fontsize=8)


def animar(Modelo, parametros, pasos=300, cada=2, archivo=None):
    """ Animación con slider (como la del ejemplo de AgentPy) de `pasos` pasos, dibujando uno de
    cada `cada`. Devuelve el HTML; si se da `archivo`, también lo guarda como página web. """
    import matplotlib.pyplot as plt
    from matplotlib import animation

    m = Modelo(dict(parametros, steps=pasos))
    m.sim_setup()
    fig, ax = plt.subplots(figsize=(7, 7))

    def cuadro(i):
        if i > 0:
            for _ in range(cada):
                if m.running:
                    m.sim_step()
        cuadro_parvada(m, ax)

    anim = animation.FuncAnimation(fig, cuadro, frames=pasos // cada + 1, interval=50)
    html = anim.to_jshtml(fps=20)
    plt.close(fig)
    if archivo:
        with open(archivo, 'w') as f:
            f.write(f'<html><head><meta charset="utf-8"><title>Flocking</title></head><body>{html}</body></html>')
    return html

# ============================================================================
# 6. KPIs y registro de datos (sección 5)
# ============================================================================

def indicadores(m):
    """ KPIs del paso actual y la máscara de aves aisladas.

    Los indicadores de estructura y rapidez se calculan siempre. Los de energía,
    dirección preferida y roles se agregan solo si el modelo tiene esos
    mecanismos: agentes con `energy` (Etapa 3), modelo con `goal_dir` (Etapa 2)
    y agentes con `role` (Etapa 2). """
    d = distancias_toroidales(m)
    np.fill_diagonal(d, np.inf)
    kpi = metricas_parvada(m, d)

    agentes = list(m.agents)
    rumbos = np.array([a.heading for a in agentes])
    rapidez = np.array([a.speed for a in agentes]) / m.p.dt  # m/s
    aislado = d.min(axis=1) >= m.p.outer_radius
    kpi['frac_aislados'] = aislado.mean()
    kpi['rapidez_media'] = rapidez.mean()
    kpi['rapidez_std'] = rapidez.std()

    tiene_energia = hasattr(agentes[0], 'energy')
    if tiene_energia:
        energia = np.array([a.energy for a in agentes])
        kpi['energia_media'] = energia.mean()
        kpi['energia_min'] = energia.min()
        kpi['energia_p10'] = np.percentile(energia, 10)
        fatigado = energia < m.p.get('e_crit', 0)
        kpi['frac_fatigados'] = fatigado.mean()
        # ¿Las aves fatigadas siguen con la parvada? Fracción de cada grupo en el grupo mayor
        _, etiquetas = connected_components(csr_matrix(d < m.p.outer_radius), directed=False)
        en_mayor = etiquetas == np.bincount(etiquetas).argmax()
        kpi['fatigados_en_grupo_mayor'] = en_mayor[fatigado].mean() if fatigado.any() else np.nan
        kpi['descansados_en_grupo_mayor'] = en_mayor[~fatigado].mean() if (~fatigado).any() else np.nan
        kpi['rapidez_fatigados'] = rapidez[fatigado].mean() if fatigado.any() else np.nan
        kpi['rapidez_descansados'] = rapidez[~fatigado].mean() if (~fatigado).any() else np.nan

    tiene_direccion = hasattr(m, 'goal_dir')
    if tiene_direccion:
        precision = rumbos @ m.goal_dir  # Coseno entre cada rumbo y la dirección preferida
        kpi['precision_rumbo'] = precision.mean()

    if hasattr(agentes[0], 'role'):
        roles = np.array([a.role for a in agentes])
        for r in np.unique(roles):
            s = roles == r
            kpi[f'polarizacion_{r}'] = np.linalg.norm(rumbos[s].mean(axis=0))
            kpi[f'aislados_{r}'] = aislado[s].mean()
            kpi[f'rapidez_{r}'] = rapidez[s].mean()
            if tiene_energia:
                kpi[f'energia_{r}'] = energia[s].mean()
            if tiene_direccion:
                kpi[f'precision_{r}'] = precision[s].mean()

    return kpi, aislado

class RegistroKPI:
    """ Agrega a un modelo el registro de KPIs en cada paso y, al final, los valores
    acumulados de cada agente. Se combina con cualquier modelo de la Parte III:

        class ModeloXKPI(RegistroKPI, ModeloX):
            pass
    """

    def setup(self):
        super().setup()
        for a in self.agents:
            a.distancia_total = 0.0  # m recorridos
            a.giro_total = 0.0       # grados girados, en valor absoluto
            a.pasos_aislado = 0      # pasos sin otra ave a menos de outer_radius

    def update(self):
        super().update()
        kpi, aislado = indicadores(self)
        for nombre, valor in kpi.items():
            self.record(nombre, valor)
        if self.t > 0:
            for a, solo in zip(self.agents, aislado):
                a.distancia_total += a.speed
                a.giro_total += abs(np.degrees(a.turn))
                a.pasos_aislado += int(solo)

    def end(self):
        super().end()
        por_agente = ['distancia_total', 'giro_total', 'pasos_aislado']
        por_agente += [v for v in ('role', 'energy') if hasattr(self.agents[0], v)]
        self.agents.record(por_agente)


class ModeloReferenciaKPI(RegistroKPI, ModeloReferencia):
    """ Modelo de referencia con registro de KPIs. """


class ModeloPercepcionKPI(RegistroKPI, ModeloPercepcion):
    """ Etapa 1 con registro de KPIs. """


class ModeloRolesKPI(RegistroKPI, ModeloRoles):
    """ Etapa 2 con registro de KPIs. """


class ModeloEnergiaKPI(RegistroKPI, ModeloEnergia):
    """ Etapa 3 con registro de KPIs. """

# ============================================================================
# 7. Calibración (sección 4.1) y experimentos (sección 6)
# ============================================================================

def calibrar(cambios):
    """ Corre 20 réplicas por configuración y promedia las métricas de los pasos 500 a 1000. """
    muestra = ap.Sample(dict(parametros_ref, **cambios))
    exp = ap.Experiment(ModeloReferenciaConDatos, muestra, iterations=20, record=True, randomize=True)
    variables = exp.run(n_jobs=-1, verbose=0).arrange_variables()
    barrido = [k for k, v in cambios.items() if isinstance(v, ap.Values)][0]
    por_corrida = (variables[variables.t >= 500]
                   .groupby([barrido, 'iteration'])
                   [['dist_vecino', 'frac_colision', 'polarizacion', 'frac_grupo_mayor', 'n_grupos']]
                   .mean())
    return variables, por_corrida

CARPETA_RESULTADOS = 'resultados'
RECALCULAR = False  # True: vuelve a correr los experimentos aunque haya resultados guardados
REPLICAS = 20
INICIO_PROMEDIO = 500  # Los KPIs por corrida se promedian entre este paso y el final

SERIES = ['polarizacion', 'dist_vecino', 'n_grupos', 'frac_grupo_mayor', 'frac_aislados', 'rapidez_media',
          'precision_rumbo', 'energia_media', 'frac_fatigados', 'fatigados_en_grupo_mayor', 'descansados_en_grupo_mayor']


def correr_experimento(nombre, configuraciones, Modelo=None, replicas=None):
    """ Corre cada configuración con `replicas` réplicas, o carga el resultado guardado.

    `configuraciones` es un diccionario {etiqueta: parámetros}; los parámetros pueden
    incluir `ap.Values` para barrer valores. Todas las configuraciones usan las mismas
    semillas por réplica, para que las diferencias no se deban al azar de la condición
    inicial. Devuelve:
    - `resumen`: una fila por corrida, con los KPIs promediados entre INICIO_PROMEDIO y
      el final, y los valores finales por ave promediados por rol (prefijo `final_`).
    - `series`: la media por paso de cada configuración. """
    Modelo = Modelo or ModeloEnergiaKPI
    replicas = replicas or REPLICAS
    ruta = {k: os.path.join(CARPETA_RESULTADOS, f'{nombre}_{k}.csv') for k in ('resumen', 'series')}
    if not RECALCULAR and all(os.path.exists(r) for r in ruta.values()):
        return pd.read_csv(ruta['resumen']), pd.read_csv(ruta['series'])

    resumenes, series = [], []
    for etiqueta, parametros in configuraciones.items():
        barridos = [k for k, v in parametros.items() if isinstance(v, ap.Values)]
        muestra = ap.Sample(parametros) if barridos else ap.Sample(parametros, n=1)
        salida = ap.Experiment(Modelo, muestra, iterations=replicas, record=True, randomize=True).run(n_jobs=-1, verbose=0)
        valores = salida.parameters.sample.reset_index()[['sample_id'] + barridos] if barridos else None

        kpi = salida.variables[Modelo.__name__].reset_index()
        aves = salida.variables[Modelo.agent_class.__name__].reset_index()
        for tabla in (kpi, aves):
            if 'sample_id' not in tabla:
                tabla['sample_id'] = 0  # Sin barrido, agentpy no incluye sample_id
        por_corrida = kpi[kpi.t >= INICIO_PROMEDIO].drop(columns='t').groupby(['sample_id', 'iteration']).mean()
        if 'role' not in aves:
            aves['role'] = 'todas'
        if 'energy' in aves:
            aves['fatigada'] = aves['energy'] < parametros.get('e_crit', 0)
        columnas = [c for c in ('energy', 'fatigada', 'distancia_total', 'giro_total', 'pasos_aislado') if c in aves]
        por_rol = aves.groupby(['sample_id', 'iteration', 'role'])[columnas].mean().unstack('role')
        por_rol.columns = [f'final_{c}_{r}' for c, r in por_rol.columns]
        resumen = por_corrida.join(por_rol).reset_index()
        resumen.insert(0, 'config', etiqueta)

        serie = kpi.groupby(['sample_id', 't'])[[c for c in SERIES if c in kpi]].mean().reset_index()
        serie.insert(0, 'config', etiqueta)
        if barridos:
            resumen = resumen.merge(valores, on='sample_id')
            serie = serie.merge(valores, on='sample_id')
        resumenes.append(resumen)
        series.append(serie)

    resumen, serie = pd.concat(resumenes, ignore_index=True), pd.concat(series, ignore_index=True)
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    resumen.to_csv(ruta['resumen'], index=False)
    serie.to_csv(ruta['series'], index=False)
    return resumen, serie


def media_ic(resumen, por, columnas):
    """ Media e intervalo de confianza del 95% (t de Student) entre réplicas. """
    g = resumen.groupby(por, sort=False)[columnas]
    media, n = g.mean(), g.count()
    mitad = g.std() / np.sqrt(n) * stats.t.ppf(0.975, n - 1)
    return media, mitad

# Configuraciones de los experimentos E1 a E6
percepcion_referencia = dict(n_neighbors=None, vision_angle=360, vision_radius=10)

configs_e1 = {
    'Referencia': parametros_ref,
    '+ Percepción': parametros_percepcion,
    '+ Roles': parametros_roles,
    '+ Energía (completo)': parametros_energia,
    'Solo roles': dict(parametros_roles, **percepcion_referencia),
    'Solo energía': dict(parametros_energia, **percepcion_referencia, frac_lider=0),
}

configs_e2 = {'Percepción': dict(parametros_percepcion, n_neighbors=ap.Values(3, 7, 15, 30),
                                 vision_angle=ap.Values(360, 300, 240))}

configs_e3 = {'Liderazgo': dict(parametros_energia, frac_lider=ap.Values(0.02, 0.05, 0.1, 0.2),
                                goal_weight=ap.Values(0.02, 0.05, 0.1, 0.2))}

configs_e4 = {'Rezagados': dict(parametros_energia, cohesion_strength=ap.Values(0.0025, 0.005, 0.01, 0.02),
                                fatigue_scale=ap.Values(0.004, 0.008, 0.016))}

configs_e5 = {'Recuperación': dict(parametros_energia, recovery_cv=ap.Values(0.0, 0.1, 0.2))}

variaciones = {
    'inner_radius': None, 'separation_strength': None, 'cohesion_strength': None, 'alignment_strength': None,
    'n_neighbors': (6, 8), 'vision_angle': None, 'goal_weight': None, 'frac_lider': None, 'frac_juvenil': None,
    'fatigue_scale': None, 'recovery_mean': None, 'e_crit': None, 'max_load_factor': None,
}
configs_e6 = {'Base': parametros_energia}
for parametro, valores in variaciones.items():
    base = parametros_energia[parametro]
    bajo, alto = valores or (base * 0.8, base * 1.2)
    configs_e6[f'{parametro} −20%'] = dict(parametros_energia, **{parametro: bajo})
    configs_e6[f'{parametro} +20%'] = dict(parametros_energia, **{parametro: alto})

EXPERIMENTOS = {
    'e1_ablacion': configs_e1,
    'e2_percepcion': configs_e2,
    'e3_liderazgo': configs_e3,
    'e4_rezagados': configs_e4,
    'e5_recuperacion': configs_e5,
    'e6_sensibilidad': configs_e6,
}


def correr_experimentos():
    """ Corre (o carga de `resultados/`) los experimentos E1 a E6. Devuelve {nombre: (resumen, series)}. """
    return {nombre: correr_experimento(nombre, configs) for nombre, configs in EXPERIMENTOS.items()}

# ============================================================================
# 8. Pruebas de verificación (sección 7.1)
# ============================================================================

def ejecutar_pruebas():
    """ Corre las pruebas de verificación del código. Devuelve una tabla con el resultado de cada una. """
    pruebas = []

    def prueba(grupo, nombre, condicion):
        """ Registra el resultado de una prueba. """
        pruebas.append({'grupo': grupo, 'prueba': nombre, 'resultado': 'pasa' if condicion else 'FALLA'})


    def kpis_de(Modelo, parametros, pasos=200):
        """ KPIs por paso de una corrida corta. """
        return Modelo(dict(parametros, steps=pasos)).run(display=False).variables[Modelo.__name__]


    # 1. Geometría del torus
    d = desplazamiento_toroidal(np.array([1., 1.]), np.array([199., 199.]), 200)
    prueba('Torus', 'El desplazamiento más corto cruza el borde', np.allclose(d, [-2, -2]))
    m = ModeloReferencia(dict(parametros_ref, population=2))
    m.sim_setup()
    a, b = m.agents
    m.space.move_to(a, np.array([0.5, 100.]))
    m.space.move_to(b, np.array([199.5, 100.]))
    prueba('Torus', 'Un vecino al otro lado del borde se detecta', len(a.space.neighbors(a, distance=2)) == 1)

    # 2. Cada etapa, con sus parámetros nuevos por defecto, reproduce exactamente a la anterior
    for Modelo, Anterior, p in [(ModeloPercepcionKPI, ModeloReferenciaKPI, parametros_ref),
                                (ModeloRolesKPI, ModeloPercepcionKPI, parametros_percepcion),
                                (ModeloEnergiaKPI, ModeloRolesKPI, parametros_roles)]:
        nueva, previa = kpis_de(Modelo, p), kpis_de(Anterior, p)
        prueba('Equivalencia', f'{Modelo.__name__[:-3]} con valores por defecto = {Anterior.__name__[:-3]}',
               nueva[previa.columns].equals(previa))

    # 3. Percepción: los k vecinos visibles coinciden con un cálculo de fuerza bruta
    p = parametros_percepcion
    m = ModeloPercepcion(p)
    m.sim_setup()
    for _ in range(100):
        m.sim_step()
    pos = np.array(list(m.space.positions.values()))
    correctos = 0
    for ave in m.space.positions:
        disp = desplazamiento_toroidal(ave.pos, pos, p['size'])
        dist = np.linalg.norm(disp, axis=1)
        visible = ((dist > 0) & (dist <= p['vision_radius'])
                   & (disp @ ave.heading / np.maximum(dist, 1e-12) >= np.cos(np.radians(p['vision_angle'] / 2))))
        cand = np.where(visible)[0]
        esperados = np.sort(dist[cand[np.argsort(dist[cand])[:p['n_neighbors']]]])
        obtenidos = np.sort(np.linalg.norm(ave.vecinos()[0], axis=1))
        correctos += np.allclose(esperados, obtenidos)
    prueba('Percepción', f'Los {p["n_neighbors"]} vecinos visibles coinciden con fuerza bruta en todas las aves', correctos == len(pos))

    for angulo, esperado in [(360, 1), (300, 0)]:
        m = ModeloPercepcion(dict(p, population=2, vision_angle=angulo))
        m.sim_setup()
        a, b = m.agents
        a.heading = np.array([1., 0.])
        m.space.move_to(a, np.array([100., 100.]))
        m.space.move_to(b, np.array([98., 100.]))
        prueba('Percepción', f'Un ave 2 m detrás se percibe con {angulo}°: {"sí" if esperado else "no"}', len(a.vecinos()[0]) == esperado)

    # 4. Roles: proporciones exactas y parámetros por rol
    for n, fl, fj in [(200, 0.10, 0.45), (7, 1 / 3, 1 / 3), (150, 0.02, 0.5)]:
        m = ModeloRoles(dict(parametros_roles, population=n, frac_lider=fl, frac_juvenil=fj))
        m.sim_setup()
        conteo = {r: sum(a.role == r for a in m.agents) for r in ROLES}
        prueba('Roles', f'{n} aves con {fl:.0%} líderes y {fj:.0%} juveniles: conteo exacto que suma {n}',
               conteo == m.conteo_roles and sum(conteo.values()) == n and abs(conteo['lider'] - fl * n) < 1)
    m = ModeloRoles(dict(parametros_roles, alignment_strength__juvenil=0.1))
    m.sim_setup()
    prueba('Roles', 'Un parámetro con sufijo __juvenil cambia solo a los juveniles',
           all((a.p.alignment_strength == 0.1) == (a.role == 'juvenil') for a in m.agents))

    # 5. Funciones de costo
    prueba('Energía', 'Costo por rapidez: 1 a 10 m/s, 0,835 a 6,3 m/s y 1,196 a 14,4 m/s',
           np.allclose([costo_rapidez(10), costo_rapidez(6.3), costo_rapidez(14.4)], [1, 0.835, 1.196], atol=1e-3))
    prueba('Energía', 'Costo por giro: 1 en vuelo recto y 4 con un giro de 2 g',
           costo_giro(10, 0, 0.1) == 1 and np.isclose(costo_giro(10, G * np.sqrt(3) * 0.01 / 1.0, 0.1), 4))
    prueba('Energía', 'Racha: un giro aislado no se encarece y el multiplicador se satura en streak_max',
           costo_racha(1, 2, 5) == 1 and np.isclose(costo_racha(200, 2, 5), 2))
    m = ModeloEnergia(parametros_energia)
    m.sim_setup()
    r = np.array([a.recovery_rate for a in m.agents])
    prueba('Energía', 'Recuperación individual con la media y la variación indicadas (±5% y ±0,03)',
           abs(r.mean() / parametros_energia['recovery_mean'] - 1) < 0.05 and abs(r.std() / r.mean() - parametros_energia['recovery_cv']) < 0.03)

    # 6. Invariantes del modelo completo durante 500 pasos
    m = ModeloEnergia(dict(parametros_energia, steps=500))
    m.sim_setup()
    a_max = G * np.sqrt(m.p.max_load_factor ** 2 - 1)
    ok_giro = ok_rapidez = ok_energia = ok_pos = ok_rumbo = True
    while m.running:
        antes = {a: (a.speed, a.v_max_actual()) for a in m.agents}
        m.sim_step()
        for a in m.agents:
            v_prev, v_max_prev = antes[a]
            ok_giro &= abs(a.turn) <= a_max * m.p.dt ** 2 / v_prev + 1e-12
            ok_rapidez &= m.p.v_min - 1e-12 <= a.speed <= v_max_prev + 1e-12
            ok_energia &= 0 <= a.energy <= 1
            ok_rumbo &= np.isclose(np.linalg.norm(a.heading), 1)
        pos = np.array(list(m.space.positions.values()))
        ok_pos &= bool((pos >= 0).all() and (pos < m.p.size).all())
    prueba('Invariantes', 'El giro por paso nunca supera el máximo para la rapidez del ave', ok_giro)
    prueba('Invariantes', 'La rapidez queda entre v_min y la rapidez máxima que permite la energía', ok_rapidez)
    prueba('Invariantes', 'La energía queda entre 0 y 1', ok_energia)
    prueba('Invariantes', 'El rumbo siempre mide 1', ok_rumbo)
    prueba('Invariantes', 'Las posiciones quedan dentro del torus', ok_pos)

    # 7. Reproducibilidad del modelo completo
    def posiciones_finales(semilla):
        m = ModeloEnergia(dict(parametros_energia, steps=200, seed=semilla))
        m.run(display=False)
        return np.array(list(m.space.positions.values()))
    prueba('Reproducibilidad', 'La misma semilla da exactamente el mismo resultado', np.array_equal(posiciones_finales(1), posiciones_finales(1)))
    prueba('Reproducibilidad', 'Una semilla distinta da otro resultado', not np.array_equal(posiciones_finales(1), posiciones_finales(2)))

    resultado_pruebas = pd.DataFrame(pruebas)
    assert (resultado_pruebas.resultado == 'pasa').all(), 'Hay pruebas que fallan'

    resultado_pruebas = pd.DataFrame(pruebas)
    assert (resultado_pruebas.resultado == 'pasa').all(), 'Hay pruebas que fallan'
    return resultado_pruebas

# ============================================================================
# Línea de comandos
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Modelo de flocking con realismo aumentado (AgentPy).')
    parser.add_argument('--animacion', nargs='?', const=300, type=int, metavar='PASOS',
                        help='genera animacion.html con una animación con slider (300 pasos por defecto) y la abre')
    parser.add_argument('--pruebas', action='store_true', help='corre las pruebas de verificación')
    parser.add_argument('--experimentos', action='store_true', help='carga o corre los experimentos E1 a E6')
    parser.add_argument('--recalcular', action='store_true', help='vuelve a correr los experimentos aunque haya resultados guardados')
    args = parser.parse_args()
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 90)

    if args.animacion:
        import webbrowser
        archivo = os.path.abspath('animacion.html')
        animar(ModeloExtendido, parametros_extendido, pasos=args.animacion, archivo=archivo)
        print(f'Animación guardada en {archivo}')
        webbrowser.open(f'file://{archivo}')
    if args.pruebas:
        print(ejecutar_pruebas().to_string())
    if args.experimentos:
        global RECALCULAR
        RECALCULAR = args.recalcular
        for nombre, (resumen, _) in correr_experimentos().items():
            columnas = [c for c in ('frac_grupo_mayor', 'polarizacion', 'dist_vecino', 'precision_rumbo', 'energia_media')
                        if c in resumen]
            print(f'\n== {nombre}: {len(resumen)} corridas')
            print(media_ic(resumen, 'config', columnas)[0].round(3).to_string())
    if not (args.animacion or args.pruebas or args.experimentos):
        salida = ModeloExtendidoKPI(parametros_extendido).run(display=False)
        kpi = salida.variables.ModeloExtendidoKPI
        print(kpi.loc[kpi.index >= INICIO_PROMEDIO, ['polarizacion', 'dist_vecino', 'n_grupos', 'frac_grupo_mayor',
                                                     'rapidez_media', 'precision_rumbo', 'energia_media',
                                                     'frac_fatigados']].mean().round(3).to_string())


class ModeloExtendidoKPI(RegistroKPI, ModeloExtendido):
    """ Modelo completo con registro de KPIs. """


if __name__ == '__main__':
    main()
