# Copyright (C) 2018 Christopher Gearhart
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
import os
import math
import time

# Blender imports
import bpy
import blf
import bgl
import bmesh
from bmesh.types import BMVert, BMEdge, BMFace
from mathutils import Vector, Matrix
from bpy_extras import view3d_utils
from mathutils.geometry import intersect_line_line, intersect_point_line, intersect_line_plane

# Addon imports
from ..addon_common.cookiecutter.cookiecutter import CookieCutter
from ..addon_common.common import ui
from ..addon_common.common.bmesh_utils import BMeshSelectState, BMeshHideState
from ..addon_common.common.maths import Point, Point2D, XForm
from ..addon_common.common.decorators import PersistentOptions
from ..functions import *
from ..functions import bgl_utils


@PersistentOptions()
class OperatorOptions:
    defaults = {
        "by": "count",
        "count": 5,
        "length": 0.5,
        "position": 9,
    }


class POINTSPICKER_OT_pick_points(CookieCutter):
    """ Pick points """
    bl_idname      = "pointspicker.pick_points"
    bl_label       = "Pick points"
    bl_description = ""
    bl_space_type  = "VIEW_3D"
    bl_region_type = "TOOLS"

    default_keymap = {
        "grab":   {"LEFTMOUSE"},
        "add":    {"SHIFT+LEFTMOUSE"},
        "remove": {"ALT+LEFTMOUSE"},
        "commit": {"RET"},
        "cancel": {"ESC"},
    }

    ################################################
    # CookieCutter Operator methods

    @classmethod
    def can_start(cls, context):
        """ Start only if editing a mesh """
        ob = context.active_object
        return ob and ob.type == "MESH"

    def start(self):
        """ ExtruCut tool is starting """
        scn = bpy.context.scene

        bpy.ops.ed.undo_push()  # push current state to undo

        self.header_text_set("PointsPicker")
        self.cursor_modal_set("CROSSHAIR")
        self.manipulator_hide()

        self.snap_type = "OBJECT"  #'SCENE' 'OBJECT'
        self.snap_ob = bpy.context.object
        self.started = False
        self.b_pts = []  #vectors representing locations of points
        self.normals = []
        self.labels = [] #strings to be drawn above points
        self.selected = -1
        self.hovered = [None, -1]

        self.grab_undo_loc = None
        self.grab_undo_no = None
        self.mouse = (None, None)

    def end_commit(self):
        """ Commit changes to mesh! """
        # scn = bpy.context.scene
        # m = bpy.data.meshes.new("points_result")
        # points_obj = bpy.data.objects.new("points_result", m)
        # bme = bmesh.new()
        # for co in self.b_pts:
        #     bme.verts.new(co)
        # bme.to_mesh(points_obj.data)
        # scn.objects.link(points_obj)
        # select(points_obj, active=True, only=True)
        pass

    def end_cancel(self):
        """ Cancel changes """
        bpy.ops.ed.undo()   # undo geometry hide

    def end(self):
        """ Restore everything, because we're done """
        self.manipulator_restore()
        self.header_text_restore()
        self.cursor_modal_restore()

    def update(self):
        """ Check if we need to update any internal data structures """
        pass

    #############################################
    # State functions

    @CookieCutter.FSM_State("main")
    def modal_main(self):

        if self.actions.pressed("add"):
            x, y = self.event.mouse_region_x, self.event.mouse_region_y
            self.click_add_point(bpy.context, x, y)
            return "main"
        if self.actions.pressed("remove"):
            self.click_remove_point()
            return "main"
        if self.actions.pressed("grab"):
            return "grab"
        if self.actions.mousemove:
            x, y = self.event.mouse_region_x, self.event.mouse_region_y
            self.hover(bpy.context, x, y)
            self.cursor_modal_set("HAND" if self.hovered[0] == "POINT" else "CROSSHAIR")

        if self.actions.pressed("commit"):
            self.done();
            return
        if self.actions.pressed("cancel"):
            self.done(cancel=True)
            return

    @CookieCutter.FSM_State("grab", "can enter")
    def can_start_grab(self):
        return self.hovered[0] == "POINT"

    @CookieCutter.FSM_State("grab", "enter")
    def start_grab(self):
        self.selected = self.hovered[1]
        self.grab_undo_loc = self.b_pts[self.selected]
        self.grab_undo_mp = self.normals[self.selected]

    @CookieCutter.FSM_State("grab")
    def modal_grab(self):
        if self.actions.released("grab"):
            self.grab_undo_loc = None
            return "main"

        if self.actions.mousemove:
            x, y = self.event.mouse_region_x, self.event.mouse_region_y
            self.grab_mouse_move(bpy.context, x, y)

    ###################################################
    # draw functions

    @CookieCutter.Draw("post2d")
    def draw_postpixel(self):
        self.draw(bpy.context)

    ###################################################
    # class variables

    # NONE!

    #############################################
    # class methods

    def glVertex(self, p : Point):
        bgl.glVertex3f(*self.xform.l2w_point(p))

    def closest_extrude_Point(self, p2D : Point2D) -> Point:
        r = self.drawing.Point2D_to_Ray(p2D)
        p,_ = intersect_line_line(
            self.extrude_pt0, self.extrude_pt1,
            r.o, r.o + r.d,
            )
        return Point(p)

    def grab_mouse_move(self,context,x,y):
        region = context.region
        rv3d = context.region_data
        coord = x, y
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        ray_target = ray_origin + (view_vector * 1000)


        hit = False
        if self.snap_type == 'SCENE':

            mx = Matrix.Identity(4) #scene ray cast returns world coords
            imx = Matrix.Identity(4)
            no_mx = imx.to_3x3().transposed()
            if bversion() < '002.077.000':
                res, obj, omx, loc, no = context.scene.ray_cast(ray_origin, ray_target)
            else:
                res, loc, no, ind, obj, omx = context.scene.ray_cast(ray_origin, view_vector)

            if res:
                hit = True

            else:
                #cast the ray into a plane a
                #perpendicular to the view dir, at the last bez point of the curve
                hit = True
                view_direction = rv3d.view_rotation * Vector((0,0,-1))
                plane_pt = self.grab_undo_loc
                loc = intersect_line_plane(ray_origin, ray_target,plane_pt, view_direction)
                no = view_direction
        elif self.snap_type == 'OBJECT':
            mx = self.snap_ob.matrix_world
            imx = mx.inverted()
            no_mx = imx.to_3x3().transposed()
            if bversion() < '002.077.000':
                loc, no, face_ind = self.snap_ob.ray_cast(imx * ray_origin, imx * ray_target)
                if face_ind != -1:
                    hit = True
            else:
                ok, loc, no, face_ind = self.snap_ob.ray_cast(imx * ray_origin, imx * ray_target - imx*ray_origin)
                if ok:
                    hit = True

        if not hit:
            self.grab_cancel()

        else:
            self.b_pts[self.selected] = mx * loc
            self.normals[self.selected] = no_mx * no

    def grab_cancel(self):
        old_co =  self.grab_undo_loc
        self.b_pts[self.selected] = old_co

    def click_add_point(self, context, x, y, label=None):
        '''
        x,y = event.mouse_region_x, event.mouse_region_y

        this will add a point into the bezier curve or
        close the curve into a cyclic curve
        '''
        region = context.region
        rv3d = context.region_data
        coord = x, y
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        ray_target = ray_origin + (view_vector * 1000)

        hit = False
        if self.snap_type == 'SCENE':
            mx = Matrix.Identity(4)  #loc is given in world loc...no need to multiply by obj matrix
            imx = Matrix.Identity(4)
            no_mx = Matrix.Identity(3)
            if bversion() < '002.077.000':
                res, obj, omx, loc, no = context.scene.ray_cast(ray_origin, ray_target)  #changed in 2.77
            else:
                res, loc, no, ind, obj, omx = context.scene.ray_cast(ray_origin, view_vector)
                iomx = omx.inverted()
                no_mx = iomx.to_3x3().transposed()
            hit = res
            if not hit:
                #cast the ray into a plane a
                #perpendicular to the view dir, at the last bez point of the curve

                view_direction = rv3d.view_rotation * Vector((0,0,-1))

                if len(self.b_pts):
                    plane_pt = self.b_pts[-1]
                else:
                    plane_pt = context.scene.cursor_location
                loc = intersect_line_plane(ray_origin, ray_target,plane_pt, view_direction)
                hit = True

        elif self.snap_type == 'OBJECT':
            mx = self.snap_ob.matrix_world
            imx = mx.inverted()
            no_mx = imx.to_3x3().transposed()

            if bversion() < '002.077.000':
                loc, no, face_ind = self.snap_ob.ray_cast(imx * ray_origin, imx * ray_target)
                if face_ind != -1:
                    hit = True
            else:
                ok, loc, no, face_ind = self.snap_ob.ray_cast(imx * ray_origin, imx * ray_target - imx*ray_origin)
                if ok:
                    hit = True

            if face_ind != -1:
                hit = True

        if not hit:
            self.selected = -1
            return

        if self.hovered[0] == None:  #adding in a new point
            self.b_pts.append(mx * loc)
            self.normals.append(no_mx * no)
            self.labels.append(label)
            return True
        if self.hovered[0] == 'POINT':
            self.selected = self.hovered[1]
            return

    def click_remove_point(self, mode='mouse'):
        if mode == 'mouse':
            if not self.hovered[0] == 'POINT': return
            self.b_pts.pop(self.hovered[1])
            self.labels.pop(self.hovered[1])
            self.normals.pop(self.hovered[1])
            self.hovered = [None, -1]
        else:
            if self.selected == -1: return
            self.b_pts.pop(self.selected)
            self.labels.pop(self.selected)
            self.normals.pop(self.selected)

    def hover(self, context, x, y):
        '''
        hovering happens in mixed 3d and screen space.  It's a mess!
        '''

        if len(self.b_pts) == 0:
            return

        region = context.region
        rv3d = context.region_data
        self.mouse = Vector((x, y))
        coord = x, y
        loc3d_reg2D = view3d_utils.location_3d_to_region_2d

        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        ray_target = ray_origin + (view_vector * 1000)

        if self.snap_type == 'OBJECT':
            mx = self.snap_ob.matrix_world
            imx = mx.inverted()

            if bversion() < '002.077.000':
                loc, no, face_ind = self.snap_ob.ray_cast(imx * ray_origin, imx * ray_target)
                if face_ind == -1:
                    #do some shit
                    pass
            else:
                res, loc, no, face_ind = self.snap_ob.ray_cast(imx * ray_origin, imx * ray_target - imx * ray_origin)
                if not res:
                    #do some shit
                    pass
        elif self.snap_type == 'SCENE':

            mx = Matrix.Identity(4) #scene ray cast returns world coords
            if bversion() < '002.077.000':
                res, obj, omx, loc, no = context.scene.ray_cast(ray_origin, ray_target)
            else:
                res, loc, no, ind, obj, omx = context.scene.ray_cast(ray_origin, view_vector)


        def dist(v):
            diff = v - Vector((x,y))
            return diff.length

        def dist3d(v3):
            if v3 == None:
                return 100000000
            delt = v3 - mx * loc
            return delt.length

        closest_3d_point = min(self.b_pts, key = dist3d)
        screen_dist = dist(loc3d_reg2D(context.region, context.space_data.region_3d, closest_3d_point))

        self.hovered = ['POINT',self.b_pts.index(closest_3d_point)] if screen_dist < 20 else [None, -1]

    def draw(self, context):
        region = context.region
        rv3d = context.space_data.region_3d
        dpi = bpy.context.user_preferences.system.dpi
        if len(self.b_pts) == 0: return
        bgl_utils.draw_3d_points(context,self.b_pts, 3)

        if self.selected != -1:
            bgl_utils.draw_3d_points(context,[self.b_pts[self.selected]], 8, color = (0,1,1,1))

        if self.hovered[0] == 'POINT':
            bgl_utils.draw_3d_points(context,[self.b_pts[self.hovered[1]]], 8, color = (0,1,0,1))

        blf.size(0, 20, dpi) #fond_id = 0
        for txt, vect in zip(self.labels, self.b_pts):
            if txt:
                vector2d = view3d_utils.location_3d_to_region_2d(region, rv3d, vect)
                blf.position(0, vector2d[0], vector2d[1], 0)
                blf.draw(0, txt) #font_id = 0

        return

    #############################################
