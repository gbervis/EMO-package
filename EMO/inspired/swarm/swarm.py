"""
    ==================================
    :mod:`swarm` -- Swarm intelligence
    ==================================
    
    This module provides standard swarm intelligence algorithms.
    
    .. Copyright 2012 Aaron Garrett

    .. This program is free software: you can redistribute it and/or modify
       it under the terms of the GNU General Public License as published by
       the Free Software Foundation, either version 3 of the License, or
       (at your option) any later version.

    .. This program is distributed in the hope that it will be useful,
       but WITHOUT ANY WARRANTY; without even the implied warranty of
       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
       GNU General Public License for more details.

    .. You should have received a copy of the GNU General Public License
       along with this program.  If not, see <http://www.gnu.org/licenses/>.
       
    .. module:: swarm
    .. moduleauthor:: Aaron Garrett <aaron.lee.garrett@gmail.com>
"""
import collections
import copy
import inspired
import math


#-----------------------------------------------------------------------
#                     PARTICLE SWARM OPTIMIZATION
#-----------------------------------------------------------------------

class PSO(inspired.ec.EvolutionaryComputation):
    """Represents a basic particle swarm optimization algorithm.
    
    This class is built upon the ``EvolutionaryComputation`` class making
    use of an external archive and maintaining the population at the previous
    timestep, rather than a velocity. This approach was outlined in 
    (Deb and Padhye, "Development of Efficient Particle Swarm Optimizers by
    Using Concepts from Evolutionary Algorithms", GECCO 2010, pp. 55--62).
    This class assumes that each candidate solution is a ``Sequence`` of
    real values.
    
    Public Attributes:
    
    - *topology* -- the neighborhood topology (default topologies.star_topology)
    
    Optional keyword arguments in ``evolve`` args parameter:
    
    - *inertia* -- the inertia constant to be used in the particle 
      updating (default 0.5)
    - *cognitive_rate* -- the rate at which the particle's current 
      position influences its movement (default 2.1)
    - *social_rate* -- the rate at which the particle's neighbors 
      influence its movement (default 2.1)
    
    """
    def __init__(self, random):
        inspired.ec.EvolutionaryComputation.__init__(self, random)
        self.topology = inspired.swarm.topologies.star_topology
        self._previous_population = []
        self.selector = self._swarm_selector
        self.replacer = self._swarm_replacer
        self.variator = self._swarm_variator
        self.archiver = self._swarm_archiver
        
    def _swarm_archiver(self, random, population, archive, args):
        if len(archive) == 0:
            return population[:]
        else:
            new_archive = []
            for i, (p, a) in enumerate(zip(population[:], archive[:])):
                if p < a:
                    new_archive.append(a)
                else:
                    new_archive.append(p)
            return new_archive
        
    def _swarm_variator(self, random, candidates, args):
        inertia = args.setdefault('inertia', 0.5)
        cognitive_rate = args.setdefault('cognitive_rate', 2.1)
        social_rate = args.setdefault('social_rate', 2.1)
        if len(self.archive) == 0:
            self.archive = self.population[:]
        if len(self._previous_population) == 0:
            self._previous_population = self.population[:]
        neighbors = self.topology(self._random, self.archive, args)
        offspring = []
        for x, xprev, pbest, hood in zip(self.population, 
                                         self._previous_population, 
                                         self.archive, 
                                         neighbors):
            nbest = max(hood)
            particle = []
            for xi, xpi, pbi, nbi in zip(x.candidate, xprev.candidate, 
                                         pbest.candidate, nbest.candidate):
                value = (xi + inertia * (xi - xpi) + 
                         cognitive_rate * random.random() * (pbi - xi) + 
                         social_rate * random.random() * (nbi - xi))
                particle.append(value)
            particle = self.bounder(particle, args)
            offspring.append(particle)
        return offspring
        
    def _swarm_selector(self, random, population, args):
        return population
        
    def _swarm_replacer(self, random, population, parents, offspring, args):
        self._previous_population = population[:]
        return offspring


#-----------------------------------------------------------------------
#                        ANT COLONY OPTIMIZATION
#-----------------------------------------------------------------------

class TrailComponent(inspired.ec.Individual):
    """Represents a discrete component of a trail in ant colony optimization.
    
    An trail component has an element, which is its essence (and which
    is equivalent to the candidate in the ``Individual`` parent class); 
    a value, which is its weight or cost; a pheromone level; and a
    desirability, which is a combination of the value and pheromone
    level (and which is equivalent to the fitness in the ``Individual``
    parent class). Note that the desirability (and, thus, the fitness)
    cannot be set manually. It is calculated automatically from the 
    value and pheromone level.

    Public Attributes:
    
    - *element* -- the actual interpretation of this component
    - *value* -- the value or cost of the component
    - *desirability* -- the worth of the component based on value and 
      pheromone level
    - *delta* -- the exponential contribution of the pheromone level on
      the desirability
    - *epsilon* -- the exponential contribution of the value on the 
      desirability
    - *maximize* -- Boolean value stating use of maximization
    
    """
    def __init__(self, element, value, maximize=True, delta=1, epsilon=1):
        inspired.ec.Individual.__init__(self, element, maximize)
        self._value = value
        self._pheromone = 0
        self.fitness = 0
        self.delta = delta
        self.epsilon = epsilon
    
    @property
    def element(self):
        return self.candidate
    
    @element.setter
    def element(self, val):
        self.candidate = val
    
    @property
    def value(self):
        return self._value
        
    @value.setter
    def value(self, val):
        self._value = val
        self.fitness = (self._pheromone ** self.delta + 
                        self._value ** self.epsilon)
    
    @property
    def pheromone(self):
        return self._pheromone
        
    @pheromone.setter
    def pheromone(self, val):
        self._pheromone = val
        self.fitness = self._pheromone + self._value ** self.epsilon
    
    @property
    def desirability(self):
        return self.fitness
    
    def __eq__(self, other):
        return self.candidate == other.candidate
    
    def __str__(self):
        return '({0}, {1})'.format(self.element, self.value)
        
    def __repr__(self):
        return str(self)
        

class ACS(inspired.ec.EvolutionaryComputation):
    """Represents an Ant Colony System discrete optimization algorithm.
    
    This class is built upon the ``EvolutionaryComputation`` class making
    use of an external archive. It assumes that candidate solutions are
    composed of instances of ``TrailComponent``.
    
    Public Attributes:
    
    - *components* -- the full set of discrete components for a given problem
    - *initial_pheromone* -- the initial pheromone on a trail (default 0)
    - *evaporation_rate* -- the rate of pheromone evaporation (default 0.1)
    - *learning_rate* -- the learning rate used in pheromone updates 
      (default 0.1)
    
    """
    def __init__(self, random, components):
        inspired.ec.EvolutionaryComputation.__init__(self, random)
        self.components = components
        self.evaporation_rate = 0.1
        self.initial_pheromone = 0
        self.learning_rate = 0.1
        self._variator = self._internal_variator
        self.archiver = self._internal_archiver
        self.replacer = inspired.ec.replacers.generational_replacement
        
    @property
    def variator(self):
        return self._variator
        
    @variator.setter
    def variator(self, value):
        self._variator = [self._internal_variator]
        if isinstance(value, collections.Sequence):
            self._variator.extend(value)
        else:
            self._variator.append(value)

    def _internal_variator(self, random, candidates, args):
        offspring = []
        for i in range(len(candidates)):
            offspring.append(self.generator(random, args))
        return offspring
        
    def _internal_archiver(self, random, population, archive, args):
        best = max(population)
        if len(archive) == 0:
            archive.append(best)
        else:
            arc_best = max(archive)
            if best > arc_best:
                archive.remove(arc_best)
                archive.append(best)
            else:
                best = arc_best
        for c in self.components:
            c.pheromone = ((1 - self.evaporation_rate) * c.pheromone + 
                           self.evaporation_rate * self.initial_pheromone)
        for c in self.components:
            if c in best.candidate:
                c.pheromone = ((1 - self.learning_rate) * c.pheromone + 
                               self.learning_rate * best.fitness)
        return archive
