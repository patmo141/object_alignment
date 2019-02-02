'''
Created on Aug 19, 2017

@author: Patrick
'''
import bpy
import bgl
import bmesh
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from bpy_extras import view3d_utils
from common_utilities import bversion
from mathutils.geometry import intersect_point_line, intersect_line_plane
import bgl_utils
import blf


class PointPicker(object):
    '''
    a helper class for clicking single points in 3d view
    '''
    def __init__(self,context,snap_type ='SCENE', snap_object = None):
        '''
        Simple base class for adding,deleting, transforming points in 3d space
        '''
        
        
        self.snap_type = snap_type  #'SCENE' 'OBJECT'
        self.snap_ob = snap_object
        self.started = False
        self.b_pts = []  #vectors representing locations of points
        self.normals = []
        self.labels = [] #strings to be drawn above points
        self.selected = -1
        self.hovered = [None, -1]
        
        self.grab_undo_loc = None
        self.grab_undo_no = None
        self.mouse = (None, None)
    
    def grab_initiate(self):
        if self.selected != -1:
            self.grab_undo_loc = self.b_pts[self.selected]
            self.grab_undo_mp = self.normals[self.selected]
            return True
        
        else:
            return False
    
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
        return
    
    def grab_confirm(self):
        self.grab_undo_loc = None
        return
               
    def click_add_point(self,context,x,y, label = None):
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

    def click_delete_point(self, mode = 'mouse'):
        if mode == 'mouse':
            if not self.hovered[0] == 'POINT': return
            self.b_pts.pop(self.hovered[1])
            self.labels.pop(self.hovered[1])
            self.normals.pop(self.hovered[1])
        else:
            if self.selected == -1: return
            self.b_pts.pop(self.selected)
            self.labels.pop(self.selected)
            self.normals.pop(self.selected)
                        

    def hover(self,context,x,y):
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
        
        if screen_dist  < 20:
            self.hovered = ['POINT',self.b_pts.index(closest_3d_point)]
            return
        
        
            self.hovered = [None, -1]
        
    def draw(self,context):
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
