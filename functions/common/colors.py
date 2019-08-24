# Copyright (C) 2020 Christopher Gearhart
# chris@bblanimation.com
# http://bblanimation.com/
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# System imports
import math

# Blender imports
import bpy
from mathutils import Matrix, Vector

# Module imports
# NONE!


def get_saturation_matrix(s:float):
    """ returns saturation matrix from saturation value """
    sr = (1 - s) * 0.3086  # or 0.2125
    sg = (1 - s) * 0.6094  # or 0.7154
    sb = (1 - s) * 0.0820  # or 0.0721
    return Matrix(((sr + s, sr, sr), (sg, sg + s, sg), (sb, sb, sb + s)))


def gamma_correct(rgba:list, val:float=2.0167):
    """ gamma correct color by value """
    r, g, b, a = rgba
    r = math.pow(r, val)
    g = math.pow(g, val)
    b = math.pow(b, val)
    return [r, g, b, a]


def get_average(rgba0:Vector, rgba1:Vector, weight:float):
    """ returns weighted average of two rgba values """
    return (rgba1 * weight + rgba0) / (weight + 1)
