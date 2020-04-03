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

class OBJECT_OT_icp_align(Operator):
    """Uses ICP alignment to iteratevely aligne two objects"""
    bl_idname = "object.align_icp"
    bl_label = "ICP Align"
    bl_options = {'REGISTER', 'UNDO'}

    ################################################
    # Blender Operator methods

    @classmethod
    def poll(cls, context):
        condition_1 = len(context.selected_objects) == 2
        condition_2 = context.object.type == 'MESH'
        return condition_1 and condition_2

    def execute(self, context):
        settings = get_addon_preferences()
        align_meth = settings.align_meth
        start = time.time()
        align_obj = context.object
        base_obj = [obj for obj in context.selected_objects if obj != align_obj][0]
        base_bvh = BVHTree.FromObject(base_obj, context.evaluated_depsgraph_get())
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
            context.view_layer.update()
            #context.scene.update()

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

    ################################################
