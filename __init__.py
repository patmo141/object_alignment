'''
Created on Apr 14, 2014

@author: Patrick
'''
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
import time
import bpy
from bpy.types import Operator
from bpy.props import FloatVectorProperty, StringProperty, IntProperty, BoolProperty, FloatProperty
from bpy.types import Operator, AddonPreferences
from bpy_extras.object_utils import AddObjectHelper, object_data_add
from mathutils import Vector, Matrix, Quaternion


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
    
    use_sampe = BoolProperty(
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

    def draw(self, context):
        layout = self.layout
        layout.label(text="Alignment Preferences")
        layout.prop(self, "icp_iterations")
        layout.prop(self, "redraw_frequency")
        layout.prop(self, "use_sample")
        layout.prop(self, "sample_fraction")
        layout.prop(self, "min_start")
        

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
    

def make_pairs(align_obj, base_obj, vlist, thresh, sample = 0):
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
            
        return A, B
        

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
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        self.modal_state = 'WAITING'
        
        n = len(context.window_manager.windows)
        
        windows = [window.as_pointer() for window in context.window_manager.windows]
        print('There are this many windows %i' % len(context.window_manager.windows))
        #pop up a new 3d view area.
        #Pray for now the user is in a 3d view
        bpy.ops.screen.area_dupli('INVOKE_DEFAULT')
        
        print('Now there are this many windows %i' % len(context.window_manager.windows))
        
        for window in context.window_manager.windows:
            if window.as_pointer() not in windows:
                print('found the new window')
                print('it is %i wide and %i tall' % (window.width, window.height))
                screen = window.screen
        
                #keep track of exisiting areas, because we will make a new one
                #areas = [area.as_pointer() for area in screen.areas]    
                for area in screen.areas:
                    print('area type: %s' % area.type)
                    if area.type == 'VIEW_3D':
                        
                        for region in area.regions:
                            print('region type: %s' % region.type)
                            if region.type == 'WINDOW':
                                break
                    
                        for space in area.spaces:
                            print('space type: %s' % space.type)
                            if space.type == 'VIEW_3D':
                                break
                    
                        break
                break
        
               
        override = context.copy()
        override['window'] = window
        override['screen'] = screen
        override['area'] = area
        
        print('reality check %s, %i' % (area.type, window.width))     
        ret = bpy.ops.screen.area_split(override, direction='VERTICAL', factor=0.5, mouse_x=-100, mouse_y=-100)
        print(ret)
        print('are we getting this far')
        
        return {'FINISHED'}
    
    
        if context.object:
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "No active object, could not finish")
            return {'CANCELLED'}
        
                    
class OJECT_OT_icp_align(bpy.types.Operator):
    """Uses ICP alignment to iteratevely aligne two objects"""
    bl_idname = "object.align_icp"
    bl_label = "ICP Align"

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
        group_lookup = {g.index: g.name for g in align_obj.vertex_groups}
        if 'icp_include' in align_obj.vertex_groups:
            group = group_lookup['icp_include']
            
            for v in align_obj.data.vertices:
                for g in v.groups:
                    if g.group == group:
                        vlist.append[v.index]
    
        elif 'icp_exclude' in align_obj.vertex_groups:
            group = group_lookup('icp_exclude')
            for v in align_obj.data.vertices:
                v_groups = [g.group for g in v.groups]
                if group not in v_groups:        
                    vlist.append[v.index]
                    
        #unfortunate way to do this..
        else:
            vlist = [v.index for v in align_obj.data.vertices]
        
        
        thresh = context.user_preferences.addons['object_alignment'].preferences.min_start
        sample = context.user_preferences.addons['object_alignment'].preferences.sample_fraction
        iters = context.user_preferences.addons['object_alignment'].preferences.icp_iterations
        factor = round(1/sample)
        
        
        for n in range(iters):
            (A, B) = make_pairs(align_obj, base_obj, vlist, thresh, factor)
            (R, T) = rigid_transform_3D(np.mat(A), np.mat(B))
            
            rot = Matrix(np.array(R))
            trans = Vector(T)
            quat = rot.to_quaternion()
            align_obj.location += trans
            align_obj.rotation_quaternion *= quat
            align_obj.update_tag()
            context.scene.update()
        
        time_taken = time.time() - start 
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