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
from bpy.props import *  #IntProperty etc
from bpy.types import Operator
from mathutils import Matrix
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree
# Addon imports
from ..functions import *

class OBJECT_OT_icp_align_feedback(Operator):
    """Uses ICP alignment to iteratevely aligne two objects and redraws every n iterations.  Slower but better to diagnose errors"""
    bl_idname = "object.align_icp_redraw"
    bl_label = "ICP Align Redraw"
    bl_options = {'REGISTER', 'UNDO'}

    ################################################
    # Blender Operator methods

    total_iters = 0
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
        self.snap_method = settings.snap_method
        self.start = time.time()
        self.align_obj = context.object
        self.base_obj = [obj for obj in context.selected_objects if obj != self.align_obj][0]
        
        self.matrix_change = Matrix.Identity(4)
        #get the bvh in local coords, yay
        bme = bmesh.new()
        bme.from_mesh(self.base_obj.data)
        mx = self.align_obj.matrix_world.inverted() @ self.base_obj.matrix_world
        bme.transform(mx)
        
        #base_bvh = BVHTree.FromObject(base_obj, context.evaluated_depsgraph_get())
        self.base_bvh = BVHTree.FromBMesh(bme)
        bme.free()
        
        base_kd = KDTree(len(base_obj.data.vertices))
        for i in range(0, len(base_obj.data.vertices)):
            base_kd.insert(mx @ base_obj.data.vertices[i].co, i)
        base_kd.balance()
        self.base_kd = base_kd
        
        #self.base_bvh = BVHTree.FromObject(self.base_obj,  context.evaluated_depsgraph_get())
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


        if self.self.snap_method == 'BVH':
            (A, B, self.d_stats) = make_pairs(self.align_obj, self.base_obj, self.base_bvh, self.vlist, self.thresh, self.sample_factor, calc_stats = self.use_target)
        else:
            (A, B, self.d_stats) = make_pairs_kd(self.align_obj, self.base_obj, self.base_kd, self.vlist, self.thresh, self.sample_factor, calc_stats = self.use_target)

        if self.align_meth == '0': #rigid transform
            M = affine_matrix_from_points(A, B, shear=False, scale=False, usesvd=True)
        elif self.align_meth == '1': # rot, loc, scale
            M = affine_matrix_from_points(A, B, shear=False, scale=True, usesvd=True)

        new_mat = Matrix.Identity(4)
        for y in range(0,4):
            for z in range(0,4):
                new_mat[y][z] = M[y][z]

        self.align_obj.data.transform(new_mat)
        self.matrix_change = new_mat @ self.matrix_change
        #self.align_obj.matrix_world = self.align_obj.matrix_world @ new_mat
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
        
        imx =self.matrix_change.inverted()
        self.align_obj.data.transform(imx)
        self.align_obj.matrix_world = self.align_obj.matrix_world @ self.matrix_change
        
        
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
    
    
class OBJECT_OT_icp_align_feedback_custom(Operator):
    """Uses ICP alignment to iteratevely aligne two objects and redraws every n iterations.  Slower but better to diagnose errors"""
    bl_idname = "object.align_icp_feedback_custom"
    bl_label = "ICP Align Feedback Custom"
    bl_options = {'REGISTER', 'UNDO'}

    ################################################
    # Blender Operator methods

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
    use_target : BoolProperty(name = 'Use Target Precision', default = True, description = 'Use convergence to stop iterations')
    
    
    
    target_verts_spacing : FloatProperty(name = 'Target Vert Spacing', default = .25, min = .1, max = 10, description = 'Target vert spacing when up-sampling sparse objects')  #1 vert per / (.25**2)  eg 250 micron scan rsoultion
    
    msd : FloatProperty(name = 'Minimum Starting Distance', default = .25, min = .01, max = 2.0)
    max_iters : IntProperty(name = 'Maximum Iterations', default = 50, min = 1, max = 100, description = 'Number of itersations')
    redraw_iters = IntProperty(name = 'Iterations Between Redraw', default = 1, min = 1, max = 20, description = 'Number of iterations beteween redraws')
    
    @classmethod
    def poll(cls, context):
        #need to take care of this stuff in start_pre
        return True
    


    #override this method
    def start_pre(self, context):
        self.align_obj = context.object
        self.base_obj = [obj for obj in context.selected_objects if obj != self.align_obj][0]
        
        
        return
    
    def end_post(self, context):
        print('aligned')
        
        #check alignment quality
        #if good -> Do something
        #if bad -> tell the user about it
        
        return
        
        
    def invoke(self,context, event):
        wm = context.window_manager
        self._timer = wm.event_timer_add(time_step = 0.01, window = context.window)
        wm.modal_handler_add(self)

        settings = get_addon_preferences()
    

        self.start = time.time()
        self.align_obj = context.object
        self.base_obj = [obj for obj in context.selected_objects if obj != self.align_obj][0]
        
        self.matrix_change = Matrix.Identity(4)
        #get the bvh in local coords, yay
        bme = bmesh.new()
        bme.from_mesh(self.base_obj.data)
        mx = self.align_obj.matrix_world.inverted() @ self.base_obj.matrix_world
        bme.transform(mx)
        
        #base_bvh = BVHTree.FromObject(base_obj, context.evaluated_depsgraph_get())
        self.base_bvh = BVHTree.FromBMesh(bme)
        bme.free()
        
        base_kd = KDTree(len(self.base_obj.data.vertices))
        for i in range(0, len(self.base_obj.data.vertices)):
            base_kd.insert(mx @ self.base_obj.data.vertices[i].co, i)
        base_kd.balance()
        self.base_kd = base_kd
        
        #self.base_bvh = BVHTree.FromObject(self.base_obj,  context.evaluated_depsgraph_get())
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
        #unfortunate way to do this..
        else:
            N =  len(self.align_obj.data.vertices)
            self.did_upsample = False
            
            if N > self.max_sample and self.use_downsample:
                print('Downsampling align verts to %i' % self.max_sample)
                
                if self.use_dynamic_sample and N > self.initial_sample:
                    
                    vlist = random.sample(range(N), self.initial_sample)
                    self.current_sample = self.initial_sample
                    
                else:
                    self.current_sample = self.max_sample
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
                
                self.did_upsample = True  #gotta rememebr we did this   
                bme_upsample.to_mesh(self.align_obj.data)
                bme_upsample.free()
                upsample_time = time.time() - upsample_start
                print('Took %f seconds to upsample low poly align object' % upsample_time)
                
                
                vlist = [v.index for v in align_obj.data.vertices]

        settings = get_addon_preferences()
        
        
        
        
        self.use_target = settings.use_target
        self.sample_factor = round(1/self.sample_fraction)
        

        self.total_iters = 0
        self.converged = False
        self.conv_t_list = [self.target_translation * 10] * 5  #store last 5 translations
        self.conv_r_list = [None] * 5

 
        return {'RUNNING_MODAL'}

    def modal(self, context, event):

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            return self.cancel(context)

        if event.type == 'TIMER':
            
            #do this many iterations, then redraw
            for i in range(0,self.redraw_iters):
                if self.total_iters <= self.max_iters and not self.converged:
                    self.iterate(context)
                    self.total_iters += 1

                else:
                    return self.finish(context)
            context.area.tag_redraw()
            
            return {'RUNNING_MODAL'}

        return {'PASS_THROUGH'}


    def execute(self, context):
        
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


        if self.snap_method == 'BVH':
            (A, B, self.d_stats) = make_pairs(self.align_obj, self.base_obj, self.base_bvh, self.vlist, self.msd, self.sample_factor, calc_stats = self.use_target)
        else:
            (A, B, self.d_stats) = make_pairs_kd(self.align_obj, self.base_obj, self.base_kd, self.vlist, self.msd, self.sample_factor, calc_stats = self.use_target)

        if self.align_meth == 'RIGID': #rigid transform
            M = affine_matrix_from_points(A, B, shear=False, scale=False, usesvd=True)
        elif self.align_meth == 'ROT_LOC_SCALE': # rot, loc, scale
            M = affine_matrix_from_points(A, B, shear=False, scale=True, usesvd=True)

        new_mat = Matrix.Identity(4)
        for y in range(0,4):
            for z in range(0,4):
                new_mat[y][z] = M[y][z]

        self.align_obj.data.transform(new_mat)
        self.matrix_change = new_mat @ self.matrix_change
        #self.align_obj.matrix_world = self.align_obj.matrix_world @ new_mat
        trans = new_mat.to_translation()
        quat = new_mat.to_quaternion()

        self.align_obj.update_tag()
        if self.d_stats:
            i = int(fmod(self.total_iters,5))
            self.conv_t_list[i] = trans.length
            self.conv_r_list[i] = abs(quat.angle)
            
            if conv_t_list[i] < 5 * self.target_translation and self.use_dynamic_sample and not self.did_upsample:
                print('Sampling more for higher precisions')
                    
                self.current_sample = int(1.5 * self.current_sample)
                
                if self.current_sample < N and self.current_sample < self.max_sample:
                    self.vlist = random.sample(range(N), current_sample)
                        
                        
                        

            if all(d < self.target_translation for d in self.conv_t_list):
                self.converged = True

                print('Converged in %s iterations' % str(self.total_iters+1))
                print('Final Translation: %f ' % self.conv_t_list[i])
                print('Final Avg Dist: %f' % self.d_stats[0])
                print('Final St Dev %f' % self.d_stats[1])
                print('Avg last 5 rotation angle: %f' % np.mean(self.conv_r_list))

    def finish(self, context):
        
        imx =self.matrix_change.inverted()
        self.align_obj.data.transform(imx)
        self.align_obj.matrix_world = self.align_obj.matrix_world @ self.matrix_change
        
        
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
        
        self.end_post(context)
        return {'FINISHED'}

    ################################################
