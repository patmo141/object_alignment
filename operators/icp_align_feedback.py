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
from numpy.ma.core import fmod

# Blender imports
import bpy
from bpy.types import Operator
from mathutils import Matrix
from mathutils.bvhtree import BVHTree

# Addon imports
from ..functions import *

class OBJECT_OT_icp_align_feedback(Operator):
    """Uses ICP alignment to iteratevely aligne two objects and redraws every n iterations.  Slower but better to diagnose errors"""
    bl_idname = "object.align_icp_redraw"
    bl_label = "ICP Align Redraw"
    bl_options = {'REGISTER', 'UNDO'}

    ################################################
    # Blender Operator methods

    @classmethod
    def poll(cls, context):
        condition_1 = len(context.selected_objects) == 2
        condition_2 = context.object and context.object.type == 'MESH'
        return condition_1 and condition_2

    def invoke(self,context, event):
        wm = context.window_manager
        self._timer = wm.event_timer_add(time_step = 0.01, window = context.window)
        wm.modal_handler_add(self)

        settings = get_addon_preferences()
        self.align_meth = settings.align_meth
        self.start = time.time()
        self.align_obj = context.object
        self.base_obj = [obj for obj in context.selected_objects if obj != self.align_obj][0]
        self.base_bvh = BVHTree.FromObject(self.base_obj, context.scene)
        self.align_obj.rotation_mode = 'QUATERNION'

        self.vlist = []
        #figure out if we need to do any inclusion/exclusion
        group_lookup = {g.name: g.index for g in self.align_obj.vertex_groups}
        if 'icp_include' in self.align_obj.vertex_groups:
            group = group_lookup['icp_include']

            for v in self.align_obj.data.vertices:
                for g in v.groups:
                    if g.group == group and g.weight > 0.9:
                        self.vlist.append(v.index)

        elif 'icp_exclude' in self.align_obj.vertex_groups:
            group = group_lookup['icp_exclude']
            for v in self.align_obj.data.vertices:
                v_groups = [g.group for g in v.groups]
                if group not in v_groups:
                    self.vlist.append(v.index)
                else:
                    for g in v.groups:
                        if g.group == group and g.weight < 0.1:
                            self.vlist.append(v.index)

        #unfortunate way to do this..
        else:
            self.vlist = [v.index for v in self.align_obj.data.vertices]
            #vlist = [range(0,len(align_obj.data.vertices]  #perhaps much smarter

        settings = get_addon_preferences()
        self.thresh = settings.min_start
        self.sample_fraction = settings.sample_fraction
        self.iters = settings.icp_iterations
        self.target_d = settings.target_d
        self.use_target = settings.use_target
        self.sample_factor = round(1/self.sample_fraction)
        self.redraw_frequency = settings.redraw_frequency

        self.total_iters = 0
        self.converged = False
        self.conv_t_list = [self.target_d * 2] * 5  #store last 5 translations
        self.conv_r_list = [None] * 5

        return {'RUNNING_MODAL'}

    def modal(self, context, event):

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            return self.cancel(context)

        if event.type == 'TIMER':
            context.area.tag_redraw()
            #do this many iterations, then redraw
            for i in range(0,self.redraw_frequency):
                if self.total_iters <= self.iters and not self.converged:
                    self.iterate(context)
                    self.total_iters += 1

                else:
                    return self.finish(context)

            return {'RUNNING_MODAL'}

        return {'PASS_THROUGH'}

    @classmethod
    def poll(cls, context):
        condition_1 = len(context.selected_objects) == 2
        conidion_2 = context.object.type == 'MESH'
        return condition_1 and condition_1

    def execute(self, context):
        settings = get_addon_preferences()
        align_meth = settings.align_meth
        start = time.time()
        align_obj = context.object
        base_obj = [obj for obj in context.selected_objects if obj != align_obj][0]
        base_bvh = BVHTree.FromObject(base_obj, context.scene)
        align_obj.rotation_mode = 'QUATERNION'

        vlist = []
        #figure out if we need to do any inclusion/exclusion
        group_lookup = {g.name: g.index for g in align_obj.vertex_groups}
        if 'icp_include' in align_obj.vertex_groups:
            group = group_lookup['icp_include']

            for v in align_obj.data.vertices:
                for g in v.groups:
                    if g.group == group and g.weight > 0.9:
                        vlist.append(v.index)

        elif 'icp_exclude' in align_obj.vertex_groups:
            group = group_lookup['icp_exclude']
            for v in align_obj.data.vertices:
                v_groups = [g.group for g in v.groups]
                if group not in v_groups:
                    vlist.append(v.index)
                else:
                    for g in v.groups:
                        if g.group == group and g.weight < 0.1:
                            vlist.append(v.index)

        #unfortunate way to do this..
        else:
            vlist = [v.index for v in align_obj.data.vertices]

        settings = get_addon_preferences()
        thresh = settings.min_start
        sample = settings.sample_fraction
        iters = settings.icp_iterations
        target_d = settings.target_d
        use_target = settings.use_target
        factor = round(1/sample)

        n = 0
        converged = False
        conv_t_list = [target_d * 2] * 5  #store last 5 translations
        conv_r_list = [None] * 5

        while n < iters  and not converged:


            (A, B, d_stats) = make_pairs(align_obj, base_obj, base_bvh, vlist, thresh, factor, calc_stats = use_target)


            if align_meth == '0': #rigid transform
                M = affine_matrix_from_points(A, B, shear=False, scale=False, usesvd=True)
            elif align_meth == '1': # rot, loc, scale
                M = affine_matrix_from_points(A, B, shear=False, scale=True, usesvd=True)

            new_mat = Matrix.Identity(4)
            for y in range(0,4):
                for z in range(0,4):
                    new_mat[y][z] = M[y][z]

            align_obj.matrix_world = align_obj.matrix_world @ new_mat
            trans = new_mat.to_translation()
            quat = new_mat.to_quaternion()

            align_obj.update_tag()
            #context.scene.update()
            context.view_layer.update()

            if d_stats:
                i = int(fmod(n,5))
                conv_t_list[i] = trans.length
                conv_r_list[i] = abs(quat.angle)

                if all(d < target_d for d in conv_t_list):
                    converged = True


                    print('Converged in %s iterations' % str(n+1))
                    print('Final Translation: %f ' % conv_t_list[i])
                    print('Final Avg Dist: %f' % d_stats[0])
                    print('Final St Dev %f' % d_stats[1])
                    print('Avg last 5 rotation angle: %f' % np.mean(conv_r_list))

            n += 1
        time_taken = time.time() - start
        if use_target and not converged:
            print('Maxed out iterations')
            print('Final Translation: %f ' % conv_t_list[i])
            print('Final Avg Dist: %f' % d_stats[0])
            print('Final St Dev %f' % d_stats[1])
            print('Avg last 5 rotation angle: %f' % np.mean(conv_r_list))

        print('Aligned obj in %f sec' % time_taken)
        return {'FINISHED'}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        return {"CANCELLED"}

    ###################################################
    # class variables

    _timer = None

    #############################################
    # class methods

    def iterate(self,context):

        (A, B, self.d_stats) = make_pairs(self.align_obj, self.base_obj, self.base_bvh, self.vlist, self.thresh, self.sample_factor, calc_stats = self.use_target)


        if self.align_meth == '0': #rigid transform
            M = affine_matrix_from_points(A, B, shear=False, scale=False, usesvd=True)
        elif self.align_meth == '1': # rot, loc, scale
            M = affine_matrix_from_points(A, B, shear=False, scale=True, usesvd=True)

        new_mat = Matrix.Identity(4)
        for y in range(0,4):
            for z in range(0,4):
                new_mat[y][z] = M[y][z]

        self.align_obj.matrix_world = self.align_obj.matrix_world @ new_mat
        trans = new_mat.to_translation()
        quat = new_mat.to_quaternion()

        self.align_obj.update_tag()
        if self.d_stats:
            i = int(fmod(self.total_iters,5))
            self.conv_t_list[i] = trans.length
            self.conv_r_list[i] = abs(quat.angle)

            if all(d < self.target_d for d in self.conv_t_list):
                self.converged = True

                print('Converged in %s iterations' % str(self.total_iters+1))
                print('Final Translation: %f ' % self.conv_t_list[i])
                print('Final Avg Dist: %f' % self.d_stats[0])
                print('Final St Dev %f' % self.d_stats[1])
                print('Avg last 5 rotation angle: %f' % np.mean(self.conv_r_list))

    def finish(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)

        time_taken = time.time() - self.start

        if self.d_stats:
            if self.use_target and not self.converged:
                print('Maxed out iterations')
                print('Final Avg Dist: %f' % self.d_stats[0])
                print('Final St Dev %f' % self.d_stats[1])
                print('Avg last 5 rotation angle: %f' % np.mean(self.conv_r_list))

        print('Aligned obj in %f sec' % time_taken)
        return {'FINISHED'}

    ################################################
