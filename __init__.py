'''
Copyright (c) 2014-2016 Patrick Moore
patrick.moore.bu@gmail.com

Created by Patrick Moore for Blender, with adaptation of works by Christoph Gohlke, Nghia Ho
    With tons of help from CoDEmanX

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.



Parts of this code are adapted from transformations.py by Christoph Gohlke
http://www.lfd.uci.edu/~gohlke/code/transformations.py

The following copyright and information is attached
# Copyright (c) 2006-2015, Christoph Gohlke
# Copyright (c) 2006-2015, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
'''



bl_info = {
    "name": "Complex Alignment",
    "author": "Patrick Moore",
    "version": (0, 1),
    "blender": (2, 6, 9),
    "location": "View3D > Tools > Alignment",
    "description": "Help align objects which have overlapping featuers",
    "warning": "",
    "wiki_url": "",
    "category": "Transform Mesh"}


import numpy as np
from numpy.ma.core import fmod
import math
import time


import bpy
import blf
import bgl
from bpy.types import Operator
from bpy.props import FloatVectorProperty, StringProperty, IntProperty, BoolProperty, FloatProperty, EnumProperty
from bpy.types import Operator, AddonPreferences
from bpy_extras.object_utils import AddObjectHelper, object_data_add
from bpy_extras import view3d_utils
from mathutils import Vector, Matrix, Quaternion
from mathutils.bvhtree import BVHTree

from .utilities import *

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0
                 
#http://www.lfd.uci.edu/~gohlke/code/transformations.py
def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
    True

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])
    
#http://www.lfd.uci.edu/~gohlke/code/transformations.py    

def vector_norm(data, axis=None, out=None):
    """Return length, i.e. Euclidean norm, of ndarray along axis.

    >>> v = numpy.random.random(3)
    >>> n = vector_norm(v)
    >>> numpy.allclose(n, numpy.linalg.norm(v))
    True
    >>> v = numpy.random.rand(6, 5, 3)
    >>> n = vector_norm(v, axis=-1)
    >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=2)))
    True
    >>> n = vector_norm(v, axis=1)
    >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=1)))
    True
    >>> v = numpy.random.rand(5, 4, 3)
    >>> n = numpy.empty((5, 3))
    >>> vector_norm(v, axis=1, out=n)
    >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=1)))
    True
    >>> vector_norm([])
    0.0
    >>> vector_norm([1])
    1.0

    """
    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    else:
        data *= data
        np.sum(data, axis=axis, out=out)
        np.sqrt(out, out)

#http://www.lfd.uci.edu/~gohlke/code/transformations.py        
def affine_matrix_from_points(v0, v1, shear=True, scale=True, usesvd=True):
    """Return affine transform matrix to register two point sets.

    v0 and v1 are shape (ndims, \*) arrays of at least ndims non-homogeneous
    coordinates, where ndims is the dimensionality of the coordinate space.

    If shear is False, a similarity transformation matrix is returned.
    If also scale is False, a rigid/Euclidean transformation matrix
    is returned.

    By default the algorithm by Hartley and Zissermann [15] is used.
    If usesvd is True, similarity and Euclidean transformation matrices
    are calculated by minimizing the weighted sum of squared deviations
    (RMSD) according to the algorithm by Kabsch [8].
    Otherwise, and if ndims is 3, the quaternion based algorithm by Horn [9]
    is used, which is slower when using this Python implementation.

    The returned matrix performs rotation, translation and uniform scaling
    (if specified).

    >>> v0 = [[0, 1031, 1031, 0], [0, 0, 1600, 1600]]
    >>> v1 = [[675, 826, 826, 677], [55, 52, 281, 277]]
    >>> affine_matrix_from_points(v0, v1)
    array([[   0.14549,    0.00062,  675.50008],
           [   0.00048,    0.14094,   53.24971],
           [   0.     ,    0.     ,    1.     ]])
    >>> T = translation_matrix(numpy.random.random(3)-0.5)
    >>> R = random_rotation_matrix(numpy.random.random(3))
    >>> S = scale_matrix(random.random())
    >>> M = concatenate_matrices(T, R, S)
    >>> v0 = (numpy.random.rand(4, 100) - 0.5) * 20
    >>> v0[3] = 1
    >>> v1 = numpy.dot(M, v0)
    >>> v0[:3] += numpy.random.normal(0, 1e-8, 300).reshape(3, -1)
    >>> M = affine_matrix_from_points(v0[:3], v1[:3])
    >>> numpy.allclose(v1, numpy.dot(M, v0))
    True

    More examples in superimposition_matrix()

    """
    v0 = np.array(v0, dtype=np.float64, copy=True)
    v1 = np.array(v1, dtype=np.float64, copy=True)

    ndims = v0.shape[0]
    if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
        print(ndims < 2)
        print(v0.shape[1] < ndims)
        print(v0.shape != v1.shape)
        
        print(ndims)
        
        raise ValueError("input arrays are of wrong shape or type")

    # move centroids to origin
    t0 = -np.mean(v0, axis=1)
    M0 = np.identity(ndims+1)
    M0[:ndims, ndims] = t0
    v0 += t0.reshape(ndims, 1)
    t1 = -np.mean(v1, axis=1)
    M1 = np.identity(ndims+1)
    M1[:ndims, ndims] = t1
    v1 += t1.reshape(ndims, 1)

    if shear:
        # Affine transformation
        A = np.concatenate((v0, v1), axis=0)
        u, s, vh = np.linalg.svd(A.T)
        vh = vh[:ndims].T
        B = vh[:ndims]
        C = vh[ndims:2*ndims]
        t = np.dot(C, np.linalg.pinv(B))
        t = np.concatenate((t, np.zeros((ndims, 1))), axis=1)
        M = np.vstack((t, ((0.0,)*ndims) + (1.0,)))
    elif usesvd or ndims != 3:
        # Rigid transformation via SVD of covariance matrix
        u, s, vh = np.linalg.svd(np.dot(v1, v0.T))
        # rotation matrix from SVD orthonormal bases
        R = np.dot(u, vh)
        if np.linalg.det(R) < 0.0:
            # R does not constitute right handed system
            R -= np.outer(u[:, ndims-1], vh[ndims-1, :]*2.0)
            s[-1] *= -1.0
        # homogeneous transformation matrix
        M = np.identity(ndims+1)
        M[:ndims, :ndims] = R
    else:
        # Rigid transformation matrix via quaternion
        # compute symmetric matrix N
        xx, yy, zz = np.sum(v0 * v1, axis=1)
        xy, yz, zx = np.sum(v0 * np.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = np.sum(v0 * np.roll(v1, -2, axis=0), axis=1)
        N = [[xx+yy+zz, 0.0,      0.0,      0.0],
             [yz-zy,    xx-yy-zz, 0.0,      0.0],
             [zx-xz,    xy+yx,    yy-xx-zz, 0.0],
             [xy-yx,    zx+xz,    yz+zy,    zz-xx-yy]]
        # quaternion: eigenvector corresponding to most positive eigenvalue
        w, V = np.linalg.eigh(N)
        q = V[:, np.argmax(w)]
        q /= vector_norm(q)  # unit quaternion
        # homogeneous transformation matrix
        M = quaternion_matrix(q)

    if scale and not shear:
        # Affine transformation; scale is ratio of RMS deviations from centroid
        v0 *= v0
        v1 *= v1
        M[:ndims, :ndims] *= math.sqrt(np.sum(v1) / np.sum(v0))

    # move centroids back
    M = np.dot(np.linalg.inv(M1), np.dot(M, M0))
    M /= M[ndims, ndims]
    return M
 
          
#modified from http://nghiaho.com/?page_id=671    
def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.transpose(AA) * BB

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2,:] *= -1
        R = Vt.T * U.T

    t = -R*centroid_A.T + centroid_B.T

    return R, t

def make_pairs(align_obj, base_obj, base_bvh, vlist, thresh, sample = 0, calc_stats = False):
    '''
    vlist is a list of vertex indices in the align object to use
    for alignment.  Will be in align_obj local space!
    '''
    mx1 = align_obj.matrix_world
    mx2 = base_obj.matrix_world
    
    imx1 = mx1.inverted()
    imx2 = mx2.inverted()
    
    verts1 = []
    verts2 = []
    if calc_stats:
        dists = []
    
    #downsample if needed
    if sample > 1:
        vlist = vlist[0::sample]
        
    if thresh > 0:
        #filter data based on an initial starting dist
        #eacg time in the routine..the limit should go down
        for vert_ind in vlist:
            
            vert = align_obj.data.vertices[vert_ind]
            #closest point for point clouds.  Local space of base obj
            co_find = imx2 * (mx1 * vert.co)
            
            #closest surface point for triangle mesh
            #this is set up for a  well modeled aligning object with
            #with a noisy or scanned base object
            if bversion() <= '002.076.00':
                #co1, normal, face_index = base_obj.closest_point_on_mesh(co_find)
                co1, n, face_index, d = base_bvh.find(co_find)
            else:
                #res, co1, normal, face_index = base_obj.closest_point_on_mesh(co_find)
                co1, n, face_index, d = base_bvh.find_nearest(co_find)
            
            dist = (mx2 * co_find - mx2 * co1).length 
            #d is now returned by bvh.find
            #dist = mx2.to_scale() * d
            if face_index != -1 and dist < thresh:
                verts1.append(vert.co)
                verts2.append(imx1 * (mx2 * co1))
                if calc_stats:
                    dists.append(dist)
        
        #later we will pre-process data to get nice data sets
        #eg...closest points after initial guess within a certain threshold
        #for now, take the verts and make them a numpy array
        A = np.zeros(shape = [3,len(verts1)])
        B = np.zeros(shape = [3,len(verts1)])
        
        for i in range(0,len(verts1)):
            V1 = verts1[i]
            V2 = verts2[i]
    
            A[0][i], A[1][i], A[2][i] = V1[0], V1[1], V1[2]
            B[0][i], B[1][i], B[2][i] = V2[0], V2[1], V2[2]
        
        if calc_stats:
            avg_dist = np.mean(dists)
            dev = np.std(dists)
            d_stats = [avg_dist, dev]
        else:
            d_stats = None
        return A, B, d_stats

class AlignmentAddonPreferences(AddonPreferences):
    # this must match the addon name, use '__package__'
    # when defining this in a submodule of a python package.
    bl_idname = __name__

    icp_iterations = IntProperty(
            name="ICP Iterations",
            default=50
            )
    
    redraw_frequency = IntProperty(
            name="Redraw Iterations",
            description = "Number of iterations between redraw, bigger = less redraw but faster completion",
            default=10)
    
    use_sample = BoolProperty(
            name = "Use Sample",
            description = "Use a sample of verts to align",
            default = False)
    
    sample_fraction = FloatProperty(
            name="Sample Fraction",
            description = "Only fraction of mesh verts for alignment. Less accurate, faster",
            default = 0.5,
            min = 0,
            max = 1)
    
    min_start = FloatProperty(
            name="Minimum Starting Dist",
            description = "Only verts closer than this distance will be used in each iteration",
            default = 0.5,
            min = 0,
            max = 20)
    
    target_d = FloatProperty(
            name="Target Translation",
            description = "If translation of 3 iterations is < target, ICP is considered sucessful",
            default = 0.01,
            min = 0,
            max = 10)
    
    use_target = BoolProperty(
            name="Use Target",
            description = "Calc alignment stats at each iteration to assess convergence. SLower per step, may result in less steps",
            default = True)
    
    align_methods =['RIGID','ROT_LOC_SCALE']#,'AFFINE']
    align_items = []
    for index, item in enumerate(align_methods):
        align_items.append((str(index), align_methods[index], str(index)))
    align_meth = EnumProperty(items = align_items, name="Alignment Method", description="Changes how picked points registration aligns object", default='0', options={'ANIMATABLE'}, update=None, get=None, set=None)

    def draw(self, context):
        layout = self.layout
        layout.label(text="Alignment Preferences")
        layout.prop(self, "icp_iterations")
        layout.prop(self, "redraw_frequency")
        layout.prop(self, "use_sample")
        layout.prop(self, "sample_fraction")
        layout.prop(self, "min_start")
        layout.prop(self, "use_target")
        layout.prop(self, "target_d")
        layout.prop(self, "align_meth")
        
class ComplexAlignmentPanel(bpy.types.Panel):
    """UI for ICP Alignment"""
    #bl_category = "Alignment"
    bl_label = "ICP Object Alignment"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'

    def draw(self, context):
        settings = get_settings()
        
        layout = self.layout

        

        row = layout.row()
        row.label(text="Alignment Tools", icon='MOD_SHRINKWRAP')

        align_obj = context.object
        if align_obj:
            row = layout.row()
            row.label(text="Align object is: " + align_obj.name)
        
        else:
            row.label(text='No Alignment Object!')
        
        if len(context.selected_objects) == 2:
            
            base_obj = [obj for obj in context.selected_objects if obj != align_obj][0]
            row = layout.row()
            row.label(text="Base object is: " + base_obj.name)
        else:
            row = layout.row()
            row.label(text="No Base object!")
        
        row = layout.row()
        row.label(text = 'Pre Processing')
        row = layout.row()    
        row.operator('object.align_include')   
        row.operator('object.align_include_clear', icon = 'X', text = '')
        
        row = layout.row()    
        row.operator('object.align_exclude')    
        row.operator('object.align_exclude_clear', icon = 'X', text = '')
        
        row = layout.row()
        row.label(text = 'Initial Alignment')
        row = layout.row()
        row.operator('object.align_picked_points')
        row.operator('screen.area_dupli', icon = 'FULLSCREEN_ENTER', text = '')
        
        row = layout.row()
        row.prop(settings, "align_meth")
        
        row = layout.row()
        row.label(text = 'Iterative Alignment')
        row = layout.row()
        row.operator('object.align_icp')
        
        row = layout.row()
        row.operator('object.align_icp_redraw')
        row = layout.row()
        row.prop(settings, 'redraw_frequency')
        row = layout.row()
        row.prop(settings, 'icp_iterations')
        row = layout.row()
        row.prop(settings, 'use_sample')
        row.prop(settings, 'sample_fraction')
        row = layout.row()
        row.prop(settings, 'min_start')
        row = layout.row()
        row.prop(settings, 'use_target')
        row.prop(settings, 'target_d')
                
class OJECT_OT_align_add_include(bpy.types.Operator):
    """Adds a vertex group and puts in weight paint mode"""
    bl_idname = "object.align_include"
    bl_label = "Paint to Include"

    @classmethod
    def poll(cls, context):
        condition1 = context.mode in {'OBJECT', 'PAINT_WEIGHT'}
        condition2 = context.active_object
        
        if condition1 and condition2:
            condition3 = context.active_object.type == 'MESH'
        else:
            condition3 = False
        return condition1 and condition2 and condition3

    def execute(self, context):
        
        if 'icp_include' not in context.object.vertex_groups:
            new_group = context.object.vertex_groups.new(name = 'icp_include')
        #remove the exclude group
        if 'icp_exclude' in context.object.vertex_groups:
            g = context.object.vertex_groups['icp_exclude']
            context.object.vertex_groups.remove(g)
        
        bpy.ops.object.vertex_group_set_active(group = 'icp_include')
            
        if context.mode != 'PAINT_WEIGHT':
            bpy.ops.object.mode_set(mode = 'WEIGHT_PAINT')
            
        return {'FINISHED'}
    
class OJECT_OT_align_include_clear(bpy.types.Operator):
    """Clears the verts from the ICP alignment include group"""
    bl_idname = "object.align_include_clear"
    bl_label = "Clear Include"

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

class OJECT_OT_align_add_exclude(bpy.types.Operator):
    """Clears the verts from the ICP alignment exclude group"""
    bl_idname = "object.align_exclude"
    bl_label = "Paint to Exclude"

    @classmethod
    def poll(cls, context):
        condition1 = context.mode in {'OBJECT', 'PAINT_WEIGHT'}
        condition2 = context.active_object
        
        if condition1 and condition2:
            condition3 = context.active_object.type == 'MESH'
        else:
            condition3 = False
        return condition1 and condition2 and condition3

    def execute(self, context):
        
        if 'icp_exclude' not in context.object.vertex_groups:
            new_group = context.object.vertex_groups.new(name = 'icp_exclude')
        #remove the exclude group
        if 'icp_include' in context.object.vertex_groups:
            g = context.object.vertex_groups['icp_include']
            context.object.vertex_groups.remove(g)
        bpy.ops.object.vertex_group_set_active(group = 'icp_exclude')
            
        if context.mode != 'PAINT_WEIGHT':
            bpy.ops.object.mode_set(mode = 'WEIGHT_PAINT')
            
        return {'FINISHED'}
    
class OJECT_OT_align_exclude_clear(bpy.types.Operator):
    """Clears the verts from the ICP alignment exclude group"""
    bl_idname = "object.align_exclude_clear"
    bl_label = "Clear Exclude"

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
        if 'icp_exclude' in context.object.vertex_groups:
            g = context.object.vertex_groups['icp_exclude']
            context.object.vertex_groups.remove(g)
            
        return {'FINISHED'}
    
def draw_callback_px(self, context):
    
    font_id = 0  # XXX, need to find out how best to get this.

    # draw some text
    y = context.region.height
    dims = blf.dimensions(0, 'A')
    
    blf.position(font_id, 10, y - 20 - dims[1], 0)
    blf.size(font_id, 20, 72)  
        
    if context.area.x == self.area_align.x:
        blf.draw(font_id, "Align: "+ self.align_msg)
        points = [self.obj_align.matrix_world * p for p in self.align_points]
        color = (1,0,0,1)
    else:
        blf.draw(font_id, "Base: " + self.base_msg)
        points = [self.obj_align.matrix_world * p for p in self.base_points]
        color = (0,1,0,1)
    
    draw_3d_points_revised(context, points, color, 4)
    
    for i, vec in enumerate(points):
        ind = str(i)
        draw_3d_text(context, font_id, ind, vec)
    
class OBJECT_OT_align_pick_points(bpy.types.Operator):
    """Algin two objects with 3 or more pair of picked poitns"""
    bl_idname = "object.align_picked_points"
    bl_label = "Align: Picked Points"

    @classmethod
    def poll(cls, context):
        condition_1 = len(context.selected_objects) == 2
        condition_2 = context.object.type == 'MESH'
        return condition_1 and condition_2

    def modal(self, context, event):
        
        tag_redraw_all_view3d()
        
        if len(self.align_points) < 3:
            self.align_msg = "Pick at least %s more pts" % str(3 - len(self.align_points))
        else:
            self.align_msg = "More points optional"
                        
        if len(self.base_points) < 3:
            self.base_msg = "Pick at last %s more pts" % str(3 - len(self.base_points))
        else:
            self.base_msg = "More points optional"
            
        
        if len(self.base_points) > 3 and len(self.align_points) > 3 and len(self.base_points) != len(self.align_points):
            
            if len(self.align_points) < len(self.base_points):
                self.align_msg = "Pick %s more pts to match" % str(len(self.base_points) - len(self.align_points))
            else:
                self.base_msg = "Pick %s more pts to match" % str(len(self.align_points) - len(self.base_points))
                
        if len(self.base_points) == len(self.align_points) and len(self.base_points) >= 3:
            self.base_msg = "Hit Enter to Align"
            self.align_msg = "Hit Enter to Align"            
    

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            
            ray_max = 10000
            
            if event.mouse_x > self.area_align.x and event.mouse_x < self.area_align.x + self.area_align.width:
                
                for reg in self.area_align.regions:
                    if reg.type == 'WINDOW':
                        region = reg
                for spc in self.area_align.spaces:
                    if spc.type == 'VIEW_3D':
                        rv3d = spc.region_3d
                
                #just transform the mouse window coords into the region coords        
                coord = (event.mouse_x - region.x, event.mouse_y - region.y)
                
                #are the cords the problem
                print('align cords: ' + str(coord))
                print(str((event.mouse_region_x, event.mouse_region_y)))
                        
                view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
                ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
                ray_target = ray_origin + (view_vector * ray_max)
            
                print('in the align object window')
                (d, (ok,hit, normal, face_index)) = ray_cast_region2d(region, rv3d, coord, self.obj_align)
                if hit:
                    print('hit! align_obj %s' % self.obj_align.name)
                    #local space of align object
                    self.align_points.append(hit)

            else:
                    
                for reg in self.area_base.regions:
                    if reg.type == 'WINDOW':
                        region = reg
                for spc in self.area_base.spaces:
                    if spc.type == 'VIEW_3D':
                        rv3d = spc.region_3d
                        
                coord = (event.mouse_x - region.x, event.mouse_y - region.y)        
                view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
                ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
                ray_target = ray_origin + (view_vector * ray_max)
                
                print('in the base object window')
                (d, (ok,hit, normal, face_index)) = ray_cast_region2d(region, rv3d, coord, self.obj_base)
                if ok:
                    print('hit! base_obj %s' % self.obj_base.name)
                    #points in local space of align object
                    self.base_points.append(self.obj_align.matrix_world.inverted() * self.obj_base.matrix_world * hit)    
            
                    
            return {'RUNNING_MODAL'}
            
        elif event.type == 'RIGHTMOUSE' and event.value == 'PRESS':
            
            if event.mouse_x > self.area_align.x and event.mouse_x < self.area_align.x + self.area_align.width:
                self.align_points.pop()
            else:
                self.base_points.pop()
            
            return {'RUNNING_MODAL'}
            
            
        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            
            return {'PASS_THROUGH'}
        
        if self.modal_state == 'NAVIGATING':
            
            if (event.type in {'MOUSEMOVE',
                               'MIDDLEMOUSE', 
                                'NUMPAD_2', 
                                'NUMPAD_4', 
                                'NUMPAD_6',
                                'NUMPAD_8', 
                                'NUMPAD_1', 
                                'NUMPAD_3', 
                                'NUMPAD_5', 
                                'NUMPAD_7',
                                'NUMPAD_9'} and event.value == 'RELEASE'):
            
                self.modal_state = 'WAITING'
                return {'PASS_THROUGH'}
            
            
        if (event.type in {'MIDDLEMOUSE', 
                                    'NUMPAD_2', 
                                    'NUMPAD_4', 
                                    'NUMPAD_6',
                                    'NUMPAD_8', 
                                    'NUMPAD_1', 
                                    'NUMPAD_3', 
                                    'NUMPAD_5', 
                                    'NUMPAD_7',
                                    'NUMPAD_9'} and event.value == 'PRESS'):
            
            self.modal_state = 'NAVIGATING'
                        
            return {'PASS_THROUGH'}
        
        elif event.type in {'ESC'}:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            return {'CANCELLED'}
        
        elif event.type == 'RET':
            
            if len(self.align_points) >= 3 and len(self.base_points) >= 3 and len(self.align_points) == len(self.base_points):
                bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
                self.de_localize(context)
                self.align_obj(context)
                
                context.scene.objects.active = self.obj_align
                self.obj_align.select = True
                self.obj_base = True
                
                return {'FINISHED'}
            
        return {'RUNNING_MODAL'}
            

        

    def de_localize(self,context):
        
        override = context.copy()
        override['area'] = self.area_align
        bpy.ops.view3d.localview(override)
        bpy.ops.view3d.view_selected(override)
        
        override['area'] = self.area_base
        bpy.ops.view3d.localview(override)
        bpy.ops.view3d.view_selected(override)
        
        #Crash Blender?       
        bpy.ops.screen.area_join(min_x=self.area_align.x,min_y=self.area_align.y, max_x=self.area_base.x, max_y=self.area_base.y)
        bpy.ops.view3d.toolshelf()
        
        #ret = bpy.ops.screen.area_join(min_x=area_base.x,min_y=area_base.y, max_x=area_align.x, max_y=area_align.y)
            
    
    def align_obj(self,context):
        
        if len(self.align_points) != len(self.base_points):
            if len(self.align_points) < len(self.base_points):
                
                self.base_points = self.base_points[0:len(self.align_points)]
            else:
                self.align_points = self.align_points[0:len(self.base_points)]
                
        A = np.zeros(shape = [3,len(self.base_points)])
        B = np.zeros(shape = [3,len(self.base_points)])
        
        for i in range(0,len(self.base_points)):
            V1 = self.align_points[i]
            V2 = self.base_points[i]
    
            A[0][i], A[1][i], A[2][i] = V1[0], V1[1], V1[2]
            B[0][i], B[1][i], B[2][i] = V2[0], V2[1], V2[2]  

        
        #test new method
        settings = get_settings()
        align_meth = settings.align_meth
        
        if align_meth == '0': #rigid transform
            M = affine_matrix_from_points(A, B, shear=False, scale=False, usesvd=True)
        elif align_meth == '1': # rot, loc, scale
            M = affine_matrix_from_points(A, B, shear=False, scale=True, usesvd=True)
        #else: #affine
            #M = affine_matrix_from_points(A, B, shear=True, scale=True, usesvd=True)
            
        
        new_mat = Matrix.Identity(4)
        for n in range(0,4):
            for m in range(0,4):
                new_mat[n][m] = M[n][m]

        #because we calced transform in local space
        #it's this easy to update the obj...
        self.obj_align.matrix_world = self.obj_align.matrix_world * new_mat

        self.obj_align.update_tag()
        context.scene.update()
        
            
    def invoke(self, context, event):
        self.modal_state = 'WAITING'
 
        self.start_time = time.time()
        #capture some mouse info to pass to the draw handler
        self.winx = event.mouse_x
        self.winy = event.mouse_y
            
        self.regx = event.mouse_region_x
        self.regy = event.mouse_region_y
        
        self.base_msg = 'Select 3 or more points'
        self.align_msg = 'Select 3 or more points'
        
        
        obj1_name = context.object.name
        obj2_name = [obj for obj in context.selected_objects if obj != context.object][0].name
        
        for ob in context.scene.objects:
            ob.select = False
        
        context.scene.objects.active = None
        
        #I did this stupid method becuase I was unsure
        #if some things were being "sticky" and not
        #remembering where they were
        obj1 = bpy.data.objects[obj1_name]
        obj2 = bpy.data.objects[obj2_name]
        
        for ob in bpy.data.objects:
            if ob.select:
                print(ob.name)
                
        screen = context.window.screen
        areas = [area.as_pointer() for area in screen.areas]
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                break 
        
        bpy.ops.view3d.toolshelf() #close the first toolshelf               
        override = context.copy()
        override['area'] = area
        
        self.area_align = area
        
        bpy.ops.screen.area_split(override, direction='VERTICAL', factor=0.5, mouse_x=-100, mouse_y=-100)
        #bpy.ops.view3d.toolshelf() #close the 2nd toolshelf
        
        context.scene.objects.active = obj1
        obj1.select = True
        obj2.select = False
        
        bpy.ops.view3d.localview(override)
        
        obj1.select = False
        context.scene.objects.active = None
        override = context.copy()
        for area in screen.areas:
            if area.as_pointer() not in areas:
                override['area'] = area
                self.area_base = area
                bpy.ops.object.select_all(action = 'DESELECT')
                context.scene.objects.active = obj2
                obj2.select = True
                override['selected_objects'] = [obj2]
                override['selected_editable_objects'] = [obj2]
                override['object'] = obj2
                override['active_object'] = obj2
                bpy.ops.view3d.localview(override)
                break
 
        
        self.obj_align = obj1
        self.obj_base = obj2
        
        #hooray, we will raycast in local view!
        self.align_points = []
        self.base_points = []
        
        context.window_manager.modal_handler_add(self)
        self._handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_px, (self, context), 'WINDOW', 'POST_PIXEL')
        return {'RUNNING_MODAL'}
                   
class OJECT_OT_icp_align(bpy.types.Operator):
    """Uses ICP alignment to iteratevely aligne two objects"""
    bl_idname = "object.align_icp"
    bl_label = "ICP Align"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        condition_1 = len(context.selected_objects) == 2
        conidion_2 = context.object.type == 'MESH'
        return condition_1 and condition_1

    def execute(self, context):
        settings = get_settings()
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
        
        settings = get_settings()
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
                
            align_obj.matrix_world = align_obj.matrix_world * new_mat
            trans = new_mat.to_translation()
            quat = new_mat.to_quaternion()
            
            align_obj.update_tag()
            context.scene.update()
        
            if d_stats:
                i = fmod(n,5)
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

class OJECT_OT_icp_align_feedback(bpy.types.Operator):
    """Uses ICP alignment to iteratevely aligne two objects and redraws every n iterations.  Slower but better to diagnose errors"""
    bl_idname = "object.align_icp_redraw"
    bl_label = "ICP Align Redraw"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None
    
    
    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        
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
    
    @classmethod
    def poll(cls, context):
        condition_1 = len(context.selected_objects) == 2
        condition_2 = context.object.type == 'MESH'
        return condition_1 and condition_2


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
            
        self.align_obj.matrix_world = self.align_obj.matrix_world * new_mat
        trans = new_mat.to_translation()
        quat = new_mat.to_quaternion()
        
        self.align_obj.update_tag()
        if self.d_stats:
            i = fmod(self.total_iters,5)
            self.conv_t_list[i] = trans.length
            self.conv_r_list[i] = abs(quat.angle)
            
            if all(d < self.target_d for d in self.conv_t_list):
                self.converged = True
                
                print('Converged in %s iterations' % str(self.total_iters+1))
                print('Final Translation: %f ' % self.conv_t_list[i])
                print('Final Avg Dist: %f' % self.d_stats[0])
                print('Final St Dev %f' % self.d_stats[1])
                print('Avg last 5 rotation angle: %f' % np.mean(self.conv_r_list))
        
                
        
        
    def invoke(self,context, event):
        
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.01, context.window)
        wm.modal_handler_add(self)
        
        
        settings = get_settings()
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
        
        settings = get_settings()
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
def register():
    bpy.utils.register_class(AlignmentAddonPreferences)
    bpy.utils.register_class(OJECT_OT_icp_align)
    bpy.utils.register_class(OJECT_OT_icp_align_feedback)
    bpy.utils.register_class(OJECT_OT_align_add_include)
    bpy.utils.register_class(OJECT_OT_align_add_exclude)
    bpy.utils.register_class(OJECT_OT_align_include_clear)
    bpy.utils.register_class(OJECT_OT_align_exclude_clear)
    bpy.utils.register_class(OBJECT_OT_align_pick_points)
    bpy.utils.register_class(ComplexAlignmentPanel)
    


def unregister():
    bpy.utils.unregister_class(AlignmentAddonPreferences)
    bpy.utils.unregister_class(OJECT_OT_icp_align)
    bpy.utils.unregister_class(OJECT_OT_icp_align_feedback)
    bpy.utils.unregister_class(OJECT_OT_align_add_include)
    bpy.utils.unregister_class(OJECT_OT_align_add_exclude)
    bpy.utils.unregister_class(OJECT_OT_align_include_clear)
    bpy.utils.unregister_class(OJECT_OT_align_exclude_clear)
    bpy.utils.unregister_class(OBJECT_OT_align_pick_points)
    bpy.utils.unregister_class(ComplexAlignmentPanel)


if __name__ == "__main__":
    register()