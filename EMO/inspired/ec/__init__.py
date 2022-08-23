"""
    ===============================================
    :mod:`ec` -- Evolutionary computation framework
    ===============================================
    
    This module provides a framework for creating evolutionary computations.
    
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
       
    .. moduleauthor:: Aaron Garrett <aaron.lee.garrett@gmail.com>
"""
from inspired.ec.ec import Bounder
from inspired.ec.ec import DEA
from inspired.ec.ec import DiscreteBounder
from inspired.ec.ec import EDA
from inspired.ec.ec import Error
from inspired.ec.ec import ES
from inspired.ec.ec import EvolutionaryComputation
from inspired.ec.ec import EvolutionExit
from inspired.ec.ec import GA
from inspired.ec.ec import Individual
from inspired.ec.ec import SA
import inspired.ec.analysis
import inspired.ec.archivers
import inspired.ec.emo
import inspired.ec.evaluators
import inspired.ec.migrators
import inspired.ec.observers
import inspired.ec.replacers
import inspired.ec.selectors
import inspired.ec.terminators
import inspired.ec.utilities
import inspired.ec.variators

__all__ = ['Bounder', 'DiscreteBounder', 'Individual', 'Error', 'EvolutionExit', 
           'EvolutionaryComputation', 'GA', 'ES', 'EDA', 'DEA', 'SA',
           'analysis', 'archivers', 'emo', 'evaluators', 'migrators', 'observers', 
           'replacers', 'selectors', 'terminators', 'utilities', 'variators']


