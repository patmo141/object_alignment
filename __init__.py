'''
Created on Apr 14, 2014

@author: Patrick
'''
from numpy.ma.core import fmod
bl_info = {
    "name": "Complex Alignment",
    "author": "Patrick Moore",
    "version": (0, 1),
    "blender": (2, 6, 0),
    "location": "View3D > Tools > Alignment",
    "description": "Help align objects which have overlapping featuers",
    "warning": "",
    "wiki_url": "",
    "category": "Transform Mesh"}


import numpy as np
import math
import time
import bpy
import blf
import bgl
from bpy.types import Operator
from bpy.props import FloatVectorProperty, StringProperty, IntProperty, BoolProperty, FloatProperty
from bpy.types import Operator, AddonPreferences
from bpy_extras.object_utils import AddObjectHelper, object_data_add
from bpy_extras import view3d_utils
from mathutils import Vector, Matrix, Quaternion

# epsilon for testing whether a number is close to zero
_EPS = numpy.finfo(float).eps * 4.0

def tag_redraw_all_view3d():
    context = bpy.context

    # Py cant access notifers
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        region.tag_redraw()
                        
def draw_3d_points(context, points, color, size):
    '''
    draw a bunch of dots
    args:
        points: a list of tuples representing x,y SCREEN coordinate eg [(10,30),(11,31),...]
        color: tuple (r,g,b,a)
        size: integer? maybe a float
    '''
    points_2d = [view3d_utils.location_3d_to_region_2d(context.region, context.space_data.region_3d, loc) for loc in points]

    bgl.glColor4f(*color)
    bgl.glPointSize(size)
    bgl.glBegin(bgl.GL_POINTS)
    for coord in points_2d:
        #TODO:  Debug this problem....perhaps loc_3d is returning points off of the screen.
        if coord:
            bgl.glVertex2f(*coord)  

    bgl.glEnd()   
    return


def draw_3d_points_revised(context, points, color, size):
    region = context.region
    region3d = context.space_data.region_3d
    
    
    region_mid_width = region.width / 2.0
    region_mid_height = region.height / 2.0
    
    perspective_matrix = region3d.perspective_matrix.copy()
    
    bgl.glColor4f(*color)
    bgl.glPointSize(size)
    bgl.glBegin(bgl.GL_POINTS)
    
    for vec in points:
    
        vec_4d = perspective_matrix * vec.to_4d()
        if vec_4d.w > 0.0:
            x = region_mid_width + region_mid_width * (vec_4d.x / vec_4d.w)
            y = region_mid_height + region_mid_height * (vec_4d.y / vec_4d.w)
            
            bgl.glVertex3f(x, y, 0)

            
    bgl.glEnd()

def draw_3d_text(context, font_id, text, vec):
    region = context.region
    region3d = context.space_data.region_3d
    
    
    region_mid_width = region.width / 2.0
    region_mid_height = region.height / 2.0
    
    perspective_matrix = region3d.perspective_matrix.copy()
    vec_4d = perspective_matrix * vec.to_4d()
    if vec_4d.w > 0.0:
        x = region_mid_width + region_mid_width * (vec_4d.x / vec_4d.w)
        y = region_mid_height + region_mid_height * (vec_4d.y / vec_4d.w)

        blf.position(font_id, x + 3.0, y - 4.0, 0.0)
        blf.draw(font_id, text)    


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


#Preferences
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
            description = "Only verts closer than this distance will be included in each iteration",
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
        

class ComplexAlignmentPanel(bpy.types.Panel):
    """UI for ICP Alignment"""
    bl_category = "Alignment"
    bl_label = "ICP Object Alignment"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'

    def draw(self, context):
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
        row.operator('object.align_include')   
        row.operator('object.align_include_clear', icon = 'X', text = '')
        
        row = layout.row()    
        row.operator('object.align_exclude')    
        row.operator('object.align_exclude_clear', icon = 'X', text = '')
        
        row = layout.row()
        row.operator('object.align_picked_points')
        row.operator('screen.area_dupli', icon = 'FULLSCREEN_ENTER', text = '')
        
        row = layout.row()
        row.operator('object.align_icp')
            

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

def obj_ray_cast(obj, matrix, ray_origin, ray_target):
    """Wrapper for ray casting that moves the ray into object space"""

    # get the ray relative to the object
    matrix_inv = matrix.inverted()
    ray_origin_obj = matrix_inv * ray_origin
    ray_target_obj = matrix_inv * ray_target

    # cast the ray
    hit, normal, face_index = obj.ray_cast(ray_origin_obj, ray_target_obj)

    if face_index != -1:
        return hit, normal, face_index
    else:
        return None, None, None
        
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
    

def make_pairs(align_obj, base_obj, vlist, thresh, sample = 0, calc_stats = False):
    '''
    vlist is a list of vertex indices in the align object to use
    for alignment
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
            #closest point for point clouds
            co_find = imx2 * (mx1 * vert.co)
            
            #closest surface point for triangle mesh
            #this is set up for a  well modeled aligning object with
            #with a noisy or scanned base object
            co1, normal, face_index = base_obj.closest_point_on_mesh(imx2 * (mx1 * vert.co))
            dist = (co_find - co1).length
            if face_index != -1 and dist < thresh:
                verts1.append(mx1 * vert.co)
                verts2.append(mx2 * co1)
                if calc_stats:
                    dists.append(dist)
        
        #later we will pre-process data to get nice data sets
        #eg...closest points after initial guess within a certain threshold
        #for now, take the verts and make them a numpy array
        A = np.zeros(shape = [len(verts1), 3])
        B = np.zeros(shape = [len(verts1), 3])
        
        for i in range(0,len(verts1)):
            V1 = verts1[i]
            V2 = verts2[i]
    
            A[i][0], A[i][1], A[i][2] = V1[0], V1[1], V1[2]
            B[i][0], B[i][1], B[i][2] = V2[0], V2[1], V2[2]
        
        if calc_stats:
            avg_dist = np.mean(dists)
            dev = np.std(dists)
            d_stats = [avg_dist, dev]
        else:
            d_stats = None
        return A, B, d_stats
        

def draw_callback_px(self, context):
    

    font_id = 0  # XXX, need to find out how best to get this.

    # draw some text
    blf.position(font_id, 10, 10, 0)
    blf.size(font_id, 20, 72)  
        
    delta = time.time() - self.start_time
    
    if context.area.x == self.area_align.x:
        blf.draw(font_id, "Align: "+ self.align_msg)
        points = self.align_points
        color = (1,0,0,1)
    else:
        blf.draw(font_id, "Base: " + self.base_msg)
        points = self.base_points
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
        conidion_2 = context.object.type == 'MESH'
        return condition_1 and condition_1

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
                hit, normal, face_index = obj_ray_cast(self.obj_align, self.obj_align.matrix_world, ray_origin, ray_target)
                
                if hit:
                    print('hit! align_obj %s' % self.obj_align.name)
                    self.align_points.append(self.obj_align.matrix_world * hit)

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
                hit, normal, face_index = obj_ray_cast(self.obj_base, self.obj_base.matrix_world, ray_origin, ray_target)
                
                if hit:
                    print('hit! base_obj %s' % self.obj_base.name)
                    self.base_points.append(self.obj_base.matrix_world * hit) #points in local space for local space drawing!      
            
                    
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
            
            if len(self.align_points) > 3 and len(self.base_points) > 3:
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
        
        override['area'] = self.area_base
        bpy.ops.view3d.localview(override)
        
    def align_obj(self,context):
        
        if len(self.align_points) != len(self.base_points):
            if len(self.align_points) < len(self.base_points):
                
                self.base_points = self.base_points[0:len(self.align_points)]
            else:
                self.align_points = self.align_points[0:len(self.base_points)]
                
        A = np.zeros(shape = [len(self.base_points), 3])
        B = np.zeros(shape = [len(self.align_points), 3])
        
        for i in range(0,len(self.base_points)):
            V1 = self.align_points[i]
            V2 = self.base_points[i]
    
            A[i][0], A[i][1], A[i][2] = V1[0], V1[1], V1[2]
            B[i][0], B[i][1], B[i][2] = V2[0], V2[1], V2[2]  
            
        
        (R, T) = rigid_transform_3D(np.mat(A), np.mat(B))
    
        rot = Matrix(np.array(R))
        trans = Vector(T)
        
        a_com = Vector((0,0,0))
        b_com = Vector((0,0,0))
        
        for i, v in enumerate(self.align_points):
            a_com += v
            b_com += self.base_points[i]
            
        a_com *= 1/len(self.align_points)
        b_com *= 1/len(self.base_points)
        print(trans.length)
        alt_trans = b_com - a_com
        print(alt_trans.length)
        
           
        quat = rot.to_quaternion()
        
        self.obj_align.location += self.obj_base.location + trans - quat * self.obj_align.matrix_world.to_translation()
        
        
        self.obj_align.rotation_mode = 'QUATERNION'
        self.obj_align.rotation_quaternion *= quat
        
        
        
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
                       
        override = context.copy()
        override['area'] = area
        
        self.area_align = area
        
        bpy.ops.screen.area_split(override, direction='VERTICAL', factor=0.5, mouse_x=-100, mouse_y=-100)
        
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
        start = time.time()
        align_obj = context.object
        base_obj = [obj for obj in context.selected_objects if obj != align_obj][0]
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
        
        
        thresh = context.user_preferences.addons['object_alignment'].preferences.min_start
        sample = context.user_preferences.addons['object_alignment'].preferences.sample_fraction
        iters = context.user_preferences.addons['object_alignment'].preferences.icp_iterations
        target_d = context.user_preferences.addons['object_alignment'].preferences.target_d
        use_target = context.user_preferences.addons['object_alignment'].preferences.use_target
        factor = round(1/sample)
        
        n = 0
        converged = False
        conv_t_list = [target_d * 2] * 5  #store last 5 translations
        conv_r_list = [None] * 5
        
        while n < iters  and not converged:
            
            
            (A, B, d_stats) = make_pairs(align_obj, base_obj, vlist, thresh, factor, calc_stats = use_target)
            (R, T) = rigid_transform_3D(np.mat(A), np.mat(B))
            
            rot = Matrix(np.array(R))
            trans = Vector(T)
            quat = rot.to_quaternion()
            align_obj.location += trans
            align_obj.rotation_quaternion *= quat
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
    
def register():
    bpy.utils.register_class(AlignmentAddonPreferences)
    bpy.utils.register_class(OJECT_OT_icp_align)
    bpy.utils.register_class(OJECT_OT_align_add_include)
    bpy.utils.register_class(OJECT_OT_align_add_exclude)
    bpy.utils.register_class(OJECT_OT_align_include_clear)
    bpy.utils.register_class(OJECT_OT_align_exclude_clear)
    bpy.utils.register_class(OBJECT_OT_align_pick_points)
    bpy.utils.register_class(ComplexAlignmentPanel)
    


def unregister():
    bpy.utils.unregister_class(AlignmentAddonPreferences)
    bpy.utils.unregister_class(OJECT_OT_icp_align)
    bpy.utils.unregister_class(ComplexAlignmentPanel)


if __name__ == "__main__":
    register()