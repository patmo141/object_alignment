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
#https://cognitivewaves.wordpress.com/opengl-vbo-shader-vao/#shader-with-vertex-buffer-object

# System imports
import time
import numpy as np

# Blender imports
import bpy
import blf
import bgl
import gpu

from gpu_extras.batch import batch_for_shader


from bpy.types import Operator
from mathutils import Matrix
from bpy_extras import view3d_utils

# Addon imports
from ..functions import *


def draw_callback_px(self, context):

    font_id = 0  # XXX, need to find out how best to get this.

    # draw some text
    y = context.region.height
    dims = blf.dimensions(0, 'A')

    #blf.position(font_id, 10, y - 20 - dims[1], 0)
    blf.position(font_id, 10, 20 + dims[1], 0)
    
    blf.size(font_id, 20, 72)

    if context.area.x == self.area_align.x:
        blf.draw(font_id, "Align: "+ self.align_msg)
        points = [self.obj_align.matrix_world @ p for p in self.align_points]
        color = (1,0,0,1)
    else:
        blf.draw(font_id, "Base: " + self.base_msg)
        points = [self.obj_align.matrix_world @ p for p in self.base_points]
        color = (0,1,0,1)

    #draw_3d_points_revised(context, points, color, 4)

    for i, vec in enumerate(points):
        ind = str(i)
        draw_3d_text(context, font_id, ind, vec)


def draw_callback_view(self, context):
    bgl.glPointSize(8)
    #print('draw view!')
    if context.area.x == self.area_align.x:
        if not self.align_shader:
            return
        
        self.align_shader.bind()
        self.align_shader.uniform_float("color", (1,0,1,1))
        self.align_batch.draw(self.align_shader)
    else:
        if not self.base_shader:
            return
        self.base_shader.bind()
        self.base_shader.uniform_float("color", (1,1,0,1))
        self.base_batch.draw(self.base_shader)
        
    bgl.glPointSize(1)
    pass
    
    

class OBJECT_OT_align_pick_points(Operator):
    """Align two objects with 3 or more pair of picked points"""
    bl_idname = "object.align_picked_points"
    bl_label = "Align: Picked Points"
    # bl_options = {"REGISTER", "UNDO"}

    ################################################
    # Blender Operator methods

    @classmethod
    def poll(cls, context):
        condition_1 = len(context.selected_objects) == 2
        condition_2 = context.object.type == 'MESH'
        return condition_1 and condition_2

    def modal(self, context, event):

        tag_redraw_areas("VIEW_3D")

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
                ray_target = ray_origin + (ray_max * view_vector)

                print('in the align object window')
                (d, (ok,hit, normal, face_index)) = ray_cast_region2d(region, rv3d, coord, self.obj_align)
                if hit:
                    print('hit! align_obj %s' % self.obj_align.name)
                    #local space of align object
                    self.align_points.append(hit)
                    self.create_batch_align()

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
                ray_target = ray_origin + (ray_max * view_vector)

                print('in the base object window')
                (d, (ok,hit, normal, face_index)) = ray_cast_region2d(region, rv3d, coord, self.obj_base)
                if ok:
                    print('hit! base_obj %s' % self.obj_base.name)
                    #points in local space of align object
                    self.base_points.append(self.obj_align.matrix_world.inverted() @ self.obj_base.matrix_world @ hit)
                    self.create_batch_base()

            return {'RUNNING_MODAL'}

        elif event.type == 'RIGHTMOUSE' and event.value == 'PRESS':

            if event.mouse_x > self.area_align.x and event.mouse_x < self.area_align.x + self.area_align.width:
                self.align_points.pop()
                self.create_batch_align()
            else:
                self.base_points.pop()
                self.create_batch_base()

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
            bpy.types.SpaceView3D.draw_handler_remove(self._2Dhandle, 'WINDOW')
            bpy.types.SpaceView3D.draw_handler_remove(self._3Dhandle, 'WINDOW')
            return {'CANCELLED'}

        elif event.type == 'RET':

            if len(self.align_points) >= 3 and len(self.base_points) >= 3 and len(self.align_points) == len(self.base_points):
                bpy.types.SpaceView3D.draw_handler_remove(self._2Dhandle, 'WINDOW')
                bpy.types.SpaceView3D.draw_handler_remove(self._3Dhandle, 'WINDOW')
                self.de_localize(context)
                self.align_obj(context)

                context.view_layer.objects.active = self.obj_align
                self.obj_align.select_set(True)
                self.obj_base = True

                return {'FINISHED'}

        return {'RUNNING_MODAL'}

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
            ob.select_set(False)

        bpy.context.view_layer.objects.active= None#context.scene.objects.active = None

        #I did this stupid method becuase I was unsure
        #if some things were being "sticky" and not
        #remembering where they were
        obj1 = bpy.data.objects[obj1_name]
        obj2 = bpy.data.objects[obj2_name]

        for ob in bpy.data.objects:
            if ob.select_set(True):
                print(ob.name)

        screen = context.window.screen
        areas = [area.as_pointer() for area in screen.areas]
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                break

        #bpy.ops.view3d.toolshelf() #close the first toolshelf
        override = context.copy()
        override['area'] = area

        self.area_align = area

        bpy.ops.screen.area_split(direction='VERTICAL', factor=0.5, cursor=(100,-100))#bpy.ops.screen.area_split(override, direction='VERTICAL', factor=0.5, mouse_x=-100, mouse_y=-100)
        #bpy.ops.view3d.toolshelf() #close the 2nd toolshelf
        

        bpy.context.view_layer.objects.active = obj1
        obj1.select_set(True)
        obj2.select_set(False)

        bpy.ops.view3d.localview(override)
        
        
        #..........Hide sidebar after area split...........................
        for A in bpy.context.screen.areas:
            if A.type == 'VIEW_3D' :
                ctx = bpy.context.copy()
                ctx['area'] = A
                bpy.ops.screen.region_toggle(ctx, region_type='UI')
#...................................................................      



        obj1.select_set(False)
        bpy.context.view_layer.objects.active = None
        override = context.copy()
        for area in screen.areas:
            if area.as_pointer() not in areas:
                override['area'] = area
                self.area_base = area
                bpy.ops.object.select_all(action = 'DESELECT')
                bpy.context.view_layer.objects.active = obj2
                obj2.select_set(True)
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

        self.base_batch = None
        self.base_shader = None
        self.align_batch = None
        self.align_shader = None
        context.window_manager.modal_handler_add(self)
        self._2Dhandle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_px, (self, context), 'WINDOW', 'POST_PIXEL')
        self._3Dhandle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_view, (self, context), 'WINDOW', 'POST_VIEW')
        
        
        return {'RUNNING_MODAL'}

    #############################################
    # class methods

    
    def create_batch_base(self):
        verts = [self.obj_align.matrix_world @ p for p in self.base_points]
        vertices = [(v.x, v.y, v.z) for v in verts]    
        self.base_shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
        self.base_batch = batch_for_shader(self.base_shader, 'POINTS', {"pos":vertices})
        
        
    def create_batch_align(self):
        verts = [self.obj_align.matrix_world @ p for p in self.align_points]
        vertices = [(v.x, v.y, v.z) for v in verts]      
        self.align_shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
        self.align_batch = batch_for_shader(self.base_shader, 'POINTS', {"pos":vertices})
        
    def de_localize(self,context):

        override = context.copy()
        override['area'] = self.area_align
        bpy.ops.view3d.localview(override)
        bpy.ops.view3d.view_selected(override)

        override['area'] = self.area_base
        bpy.ops.view3d.localview(override)
        bpy.ops.view3d.view_selected(override)

#............Crash Blender? Resolve................................
        xj = int(self.area_align.width + 1)
        yj = int(self.area_align.y + self.area_align.height / 2)
        bpy.ops.screen.area_join(cursor=(xj,yj))
#..................................................................
        #bpy.ops.view3d.toolshelf()

        bpy.ops.screen.screen_full_area()
        bpy.ops.screen.screen_full_area()
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
        settings = get_addon_preferences()
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
        self.obj_align.matrix_world = self.obj_align.matrix_world @ new_mat

        self.obj_align.update_tag()
        context.view_layer.update()
