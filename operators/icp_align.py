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
import random
import numpy as np
from numpy.ma.core import fmod

# Blender imports
import bpy
from bpy.types import Operator
from mathutils import Matrix
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

# Addon imports
from ..functions import *

from bpy.props import *

from object_print3d_utils import mesh_helpers



class OBJECT_OT_icp_align(Operator):
    """Uses ICP alignment to iteratively align two objects"""
    bl_idname = "object.align_icp"
    bl_label = "ICP Align"
    bl_options = {'REGISTER', 'UNDO'}

    ################################################
    # Blender Operator methods

    @classmethod
    def poll(cls, context):
        condition_1 = len(context.selected_objects) == 2
        conidion_2 = context.object.type == 'MESH'
        return condition_1 and condition_1



    max_sample : IntProperty(name = 'Max Sample', default = 25000, min = 100, max = 1000000, description = 'Partially sample point cloud to inrease speed')
    minimum_sample : IntProperty(name = 'Minimum Sample', default = 1000, min = 100, max = 10000, description = 'UpSample align_object point cloud to increase accuracy for low res object')
    initial_sample : IntProperty(name = 'Initial Sample', default = 10000, min = 100, max = 50000)
    use_downsample : BoolProperty(name = 'Down Sample', default = True, description = 'Use fast down sampling if needed for high poly objects')
    up_sample : BoolProperty(name = 'Upsample if Needed', default = True, description = 'Upsample the align object if n_verts < minimum_sample')
    use_dynamic_sample : BoolProperty(name = 'Dynamic Sample', default = True, description = 'Use fast down sampling for first iterations then use max sample')
    
    snap_method : EnumProperty(name = 'Snap Method', items = [('BVH', 'BVH', 'BVH'),('KD', 'KD', 'KD')], default = 'BVH')
    align_meth : EnumProperty(name = 'Align Method', items = [('RIGID', 'RIGID', 'RIGID'),('ROT_LOC_SCALE', 'ROT_LOC_SCALE', 'ROT_LOC_SCALE')], default = 'RIGID')
    
    target_translation : FloatProperty(name = 'Translation Target', default = .0001, min = .00000001, max = 1.0, step = .001)
    target_rotation : FloatProperty(name = 'Rotation Target', default = .0001, min = .00000001, max = 1.0, step = .001)
    target_verts_spacing : FloatProperty(name = 'Target Vert Spacing', default = .25, min = .1, max = 10, description = 'Target vert spacing when up-sampling sparse objects')  #1 vert per / (.25**2)  eg 250 micron scan rsoultion
    
    msd : FloatProperty(name = 'Minimum Starting Distance', default = .25, min = .01, max = 2.0)
    
    
    #use this to override things
    
    def start_pre(self, context):
        
        self.align_obj = context.object
        self.base_obj = [obj for obj in context.selected_objects if obj != self.align_obj][0]
        
        
        return
    
    
    
    def invoke(self, context, event):
        
        
        
        return context.window_manager.invoke_props_dialog(self, width = 300)
    
    
    
    def execute(self, context):
        
        
        self.start_pre(context)  #


        settings = get_addon_preferences()

        start = time.time()
        
        align_obj = self.align_obj
        base_obj = self.base_obj
        
        #get the bvh in local coords, yay
        bme = bmesh.new()
        bme.from_mesh(base_obj.data)
        mx = align_obj.matrix_world.inverted() @ base_obj.matrix_world
        bme.transform(mx)
        
        #base_bvh = BVHTree.FromObject(base_obj, context.evaluated_depsgraph_get())
        base_bvh = BVHTree.FromBMesh(bme)
        bme.free()
        
        
        base_kd = KDTree(len(base_obj.data.vertices))
        for i in range(0, len(base_obj.data.vertices)):
            base_kd.insert(mx @ base_obj.data.vertices[i].co, i)
        base_kd.balance()
        
        
        align_obj.rotation_mode = 'QUATERNION'

       
        vlist = []
        #https://blender.stackexchange.com/questions/75223/finding-vertices-in-a-vertex-group-using-blenders-python-api/75240#75240
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
            N =  len(self.align_obj.data.vertices)
            did_upsample = False
            
            if N > self.max_sample and self.use_downsample:
                print('Downsampling align verts to %i' % self.max_sample)
                
                if self.use_dynamic_sample and N > self.initial_sample:
                    
                    vlist = random.sample(range(N), self.initial_sample)
                    current_sample = self.initial_sample
                    
                else:
                    current_sample = self.max_sample
                    vlist = random.sample(range(N), self.max_sample)
            #numpy, random_choice
        
            if N < self.minimum_sample and self.up_sample:
                print('Upsampling align verts from %i to %i' % (N, self.minimum_sample))
                upsample_start = time.time()
                
                bme_align = bmesh.new()
                bme_align.from_mesh(self.align_obj.data)
                
                bme_upsample = bme_align.copy()
                bme_upsample.verts.ensure_lookup_table()
                bme_upsample.faces.ensure_lookup_table()
                area = 0
                
                non_tris = [f for f in bme_upsample.faces if len(f.verts) > 3]
                bmesh.ops.triangulate(bme_upsample, faces=non_tris, quad_method='BEAUTY', ngon_method='BEAUTY')
                
                old_verts = [v.co for v in bme_upsample.verts]  #get ready to do a foreach set
                new_verts = []

                areas = [f.calc_area() for f in bme_upsample.faces]  #not expensive to do this becaues it's low poly
                total_area = sum(areas)
                
                avg_area = total_area/len(areas)
                target_average = total_area/self.minimum_sample
                for f in bme_upsample.faces:
                    
                    if areas[f.index] < target_average: continue
                    
                    ratio = int(target_average/areas[f.index])
                    ratio = max(1, ratio)
                    perim = f.calc_perimeter()
                    numpoints = 3 * ratio
                    rand_points = mesh_helpers.bmesh_face_points_random(f, num_points=numpoints)
                    new_verts += rand_points
                    #print(rand_points)
                    
                for v in new_verts:
                    #print('adding new vert')
                    bme_upsample.verts.new(v)
                
                did_upsample = True  #gotta rememebr we did this   
                bme_upsample.to_mesh(self.align_obj.data)
                bme_upsample.free()
                upsample_time = time.time() - upsample_start
                print('Took %f seconds to upsample low poly align object' % upsample_time)
                
                
                vlist = [v.index for v in align_obj.data.vertices]

        print('vlist is %i verts' % len(vlist))
        settings = get_addon_preferences()
        thresh = settings.min_start
        sample = settings.sample_fraction
        iters = settings.icp_iterations
        target_d = settings.target_d
        use_target = settings.use_target
        snap_mode = settings.snap_method
        factor = round(1/sample)

        n = 0
        converged = False
        conv_t_list = [target_d * 2] * 5  #store last 5 translations
        conv_r_list = [None] * 5

        original_mx = align_obj.matrix_world
        matrix_change = Matrix.Identity(4)
        
        iters_start = time.time()
        while n < iters  and not converged:
            iter_start = time.time()

            if snap_mode == 'BVH':
                (A, B, d_stats) = make_pairs(align_obj, base_obj, base_bvh, vlist, thresh, calc_stats = use_target)
            else:
                (A, B, d_stats) = make_pairs_kd(align_obj, base_obj, base_kd, vlist, thresh, calc_stats = use_target)

            pair_time = time.time()
            print('  Made pairs in %f seconds' % (pair_time-iter_start))
            
            
            if self.align_meth == 'RIGID': #rigid transform
                M = affine_matrix_from_points(A, B, shear=False, scale=False, usesvd=True)
            elif self.align_meth == 'ROT_LOC_SCALE': # rot, loc, scale
                M = affine_matrix_from_points(A, B, shear=False, scale=True, usesvd=True)

            affine_time = time.time()
            print('  Affine matrix tooth %f seconds' % (affine_time - pair_time))
            
            
            new_mat = Matrix.Identity(4)
            for y in range(0,4):
                for z in range(0,4):
                    new_mat[y][z] = M[y][z]

            #align_obj.matrix_world = align_obj.matrix_world @ new_mat
            
            align_obj.data.transform(new_mat)
            matrix_change = new_mat @ matrix_change
            
            trans = new_mat.to_translation()
            quat = new_mat.to_quaternion()

            align_obj.update_tag()
            context.view_layer.update()
            #context.scene.update()

            transform_time = time.time() - affine_time
            iter_time = time.time() - iter_start
            print('  transformed data in %f seconds' % transform_time)
            
            
            #do cool dynamic sampling and stuff
            if d_stats:
                i = int(fmod(n,5))
                conv_t_list[i] = trans.length
                conv_r_list[i] = abs(quat.angle)

                print('Orietatino magnitude changes %fmm and %fradians' % (conv_t_list[i], conv_r_list[i]))
               
               
                if conv_t_list[i] < 5 * target_d and self.use_dynamic_sample and not did_upsample:
                    print('Sampling more at higher precisions')
                    
                    current_sample = int(1.5 * current_sample)
                    
                    if current_sample < N and current_sample < self.max_sample:
                        vlist = random.sample(range(N), current_sample)
                        
                        
                if all(d < target_d for d in conv_t_list):
                    converged = True


                    print('Converged in %s iterations' % str(n+1))
                    print('Final Translation: %f ' % conv_t_list[i])
                    print('Final Avg Dist: %f' % d_stats[0])
                    print('Final St Dev %f' % d_stats[1])
                    print('Avg last 5 rotation angle: %f' % np.mean(conv_r_list))
                
            print('total iteration %f seconds\n\n' % (iter_time))
            n += 1
            
        time_taken = time.time() - start
        iters_time = time.time() - iters_start
        print('Tok %f secds for %i iters' % (iters_time, n))
        print('Average time per iter %f: ' % (iters_time/n))
        
        total_rotation_change = matrix_change.to_quaternion().to_axis_angle()
        
        #put the data back to original but transform the object...OH, cache original verts and use foreach SET!
        #we do this because of speed of mesh.transform() compared to for loops over numpy lists
        #it was faster to actually transform the mesh data and get vert.co from mesh.data.verts
        #becaue we were forced to use a for loop to do the bvh.find_nearest
        imx = matrix_change.inverted()
        align_obj.data.transform(imx)
        align_obj.matrix_world = align_obj.matrix_world @ matrix_change
        
        print('Total rotational change %f' % (180*total_rotation_change[1]/math.pi))
        print('Total translational change %f' % matrix_change.to_translation().length)
        if use_target and not converged:
            print('Maxed out iterations')
            print('Final Translation: %f ' % conv_t_list[i])
            print('Final Avg Dist: %f' % d_stats[0])
            print('Final St Dev %f' % d_stats[1])
            print('Avg last 5 rotation angle: %f' % np.mean(conv_r_list))

        if N < self.minimum_sample and self.up_sample:
            #put the unbothered data back
            bme_align.to_mesh(self.align_obj.data)
            
            
        print('Aligned obj in %f sec' % time_taken)
        return {'FINISHED'}

    ################################################
