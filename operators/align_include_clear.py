# Copyright (C) 2019 Christopher Gearhart
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
# NONE!

# Blender imports
import bpy
from bpy.types import Operator

# Module imports
from ..functions import *


class OBJECT_OT_align_include_clear(Operator):
    """Clears the verts from the ICP alignment include group"""
    bl_idname = "object.align_include_clear"
    bl_label = "Clear Include"
    # bl_options = {"REGISTER", "UNDO"}

    ################################################
    # Blender Operator methods

    @classmethod
    def poll(cls, context):
        condition1 = context.mode != 'PAINT_WEIGHT'
        condition2 = context.active_object

        if condition1 and condition2:
            condition3 = context.active_object.type == 'MESH'
        else:
            condition3 = False
        return condition1 and condition2 and condition3

    def execute(self, context):
        if 'icp_include' in context.object.vertex_groups:
            g = context.object.vertex_groups['icp_include']
            context.object.vertex_groups.remove(g)
        return {'FINISHED'}

    ################################################
