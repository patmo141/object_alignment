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
import time
import numpy as np

# Blender imports
import bpy
import blf
from bpy.types import Operator
from mathutils import Vector, Matrix
from bpy_extras import view3d_utils

# Addon imports
from ..functions import *
from ..lib.points_picker.operators.points_picker import *


class OBJECT_OT_align_pick_points(VIEW3D_OT_points_picker):
    """Align two objects with 3 or more pair of picked points"""
    bl_idname = "object.align_picked_points"
    bl_label = "Align: Picked Points"
    # bl_options = {"REGISTER", "UNDO"}

    #############################################
    # overwriting functions from Points Picker submodule

    @classmethod
    def can_start(cls, context):
        """ Start only if editing a mesh """
        ob = context.active_object
        condition_1 = len(context.selected_objects) == 2
        condition_2 = ob and ob.type == 'MESH'
        return condition_1 and condition_2

    def start_post(self):
        # override default settings
        self.snap_type = "SCENE"

        # set up custom data structures
        self.align_points = list()
        self.base_points = list()
        self.align_obj = bpy.context.active_object
        self.base_obj = [obj for obj in bpy.context.selected_objects if obj != bpy.context.object][0]

        # additional UI changes
        # self.additional_ui_setup()

    def add_point_post(self, point):
        if point.source_object == self.align_obj:
            self.align_points.append(point)
        elif point.source_object == self.base_obj:
            self.base_points.append(point)
        else:
            self.b_pts.pop(self.selected)

    def can_commit(self):
        if len(self.align_points) > 2 and len(self.base_points) > 2:
            return True
        else:
            self.report({"WARNING"}, "Must pick at least 3 points per object before committing")
            return False

    def end_commit(self):
        """ Commit changes to mesh! """
        scn = bpy.context.scene
        for pt in self.b_pts:
            # self.de_localize(bpy.context)
            self.align_objects(bpy.context)
            select(self.align_obj, active=True)
        self.end_commit_post()

    def getLabel(self, idx):
        point = self.b_pts[idx]
        if point.source_object == self.align_obj:
            return "A" + str(len(self.align_points))
        elif point.source_object == self.base_obj:
            return "B" + str(len(self.base_points))

    #############################################
    # additional functions

    # def get_matrix_world_for_point(self, pt):
    #    Z = pt.surface_normal
    #    x_rand = Vector((random.random(), random.random(), random.random()))
    #    x_rand.normalize()
    #
    #    if abs(x_rand.dot(Z)) > .9:
    #        x_rand = Vector((random.random(), random.random(), random.random()))
    #        x_rand.normalize()
    #    X = x_rand - x_rand.dot(Z) * Z
    #    X.normalize()
    #
    #    Y = Z.cross(X)
    #
    #    R = Matrix.Identity(3)  #make the columns of matrix U, V, W
    #    R[0][0], R[0][1], R[0][2]  = X[0] ,Y[0],  Z[0]
    #    R[1][0], R[1][1], R[1][2]  = X[1], Y[1],  Z[1]
    #    R[2][0] ,R[2][1], R[2][2]  = X[2], Y[2],  Z[2]
    #    R = R.to_4x4()
    #
    #    if pt.label == "Replacement Point":
    #        T = Matrix.Translation(pt.location + 2 * Z)
    #    else:
    #        T = Matrix.Translation(pt.location)
    #
    #    return T * R

    def de_localize(self,context):

        override = context.copy()
        override['area'] = self.area_align
        bpy.ops.view3d.localview(override)
        bpy.ops.view3d.view_selected(override)

        override['area'] = self.area_base
        bpy.ops.view3d.localview(override)
        bpy.ops.view3d.view_selected(override)

        # Crash Blender?
        bpy.ops.screen.area_join(min_x=self.area_align.x,min_y=self.area_align.y, max_x=self.area_base.x, max_y=self.area_base.y)
        bpy.ops.view3d.toolshelf()

        # ret = bpy.ops.screen.area_join(min_x=area_base.x,min_y=area_base.y, max_x=area_align.x, max_y=area_align.y)

    def align_objects(self, context):
        scn = bpy.context.scene

        # match length of both lists to the shortest of the two
        if len(self.align_points) != len(self.base_points):
            if len(self.align_points) < len(self.base_points):
                self.base_points = self.base_points[0:len(self.align_points)]
            else:
                self.align_points = self.align_points[0:len(self.base_points)]

        A = np.zeros(shape=[3, len(self.base_points)])
        B = np.zeros(shape=[3, len(self.base_points)])

        for i in range(0, len(self.base_points)):
            V1 = self.align_points[i]
            V2 = self.base_points[i]

            A[0][i], A[1][i], A[2][i] = V1.location.x, V1.location.y, V1.location.z
            B[0][i], B[1][i], B[2][i] = V2.location.x, V2.location.y, V2.location.z


        # test new method
        align_meth = get_addon_preferences().align_meth

        if align_meth == '0': #rigid transform
            M = affine_matrix_from_points(A, B, shear=False, scale=False, usesvd=True)
        elif align_meth == '1': # rot, loc, scale
            M = affine_matrix_from_points(A, B, shear=False, scale=True, usesvd=True)
        # else: #affine
        #     M = affine_matrix_from_points(A, B, shear=True, scale=True, usesvd=True)


        new_mat = Matrix.Identity(4)
        for n in range(0,4):
            for m in range(0,4):
                new_mat[n][m] = M[n][m]

        # because we calced transform in local space
        # it's this easy to update the obj...
        self.align_obj.matrix_world = self.align_obj.matrix_world * new_mat

        self.align_obj.update_tag()
        scn.update()

    def additional_ui_setup(self):
        screen = bpy.context.window.screen
        areas = [area.as_pointer() for area in screen.areas]
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                break

        bpy.ops.view3d.toolshelf() # close the first toolshelf
        override = bpy.context.copy()
        override['area'] = area

        self.area_align = area

        bpy.ops.screen.area_split(override, direction='VERTICAL', factor=0.5, mouse_x=-100, mouse_y=-100)

        select(self.align_obj, only=True, active=True)

        bpy.ops.view3d.localview(override)

        self.align_obj.select = False
        setActiveObj(None)
        override = bpy.context.copy()
        for area in screen.areas:
            if area.as_pointer() not in areas:
                override['area'] = area
                self.area_base = area
                select(self.base_obj, only=True, active=True)
                override['selected_objects'] = [self.base_obj]
                override['selected_editable_objects'] = [self.base_obj]
                override['object'] =self.base_obj
                override['active_object'] =self.base_obj
                bpy.ops.view3d.localview(override)
                break

    #############################################
