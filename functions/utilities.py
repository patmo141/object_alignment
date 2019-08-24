# Copyright (c) 2019 Patrick Moore
# patrick.moore.bu@gmail.com
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
# NONE!

# Blender imports
import bpy
import bgl
import blf
from mathutils import Vector
from bpy_extras.view3d_utils import location_3d_to_region_2d, region_2d_to_vector_3d
from bpy_extras.view3d_utils import region_2d_to_origin_3d

# Module imports
from .common import *


#################################################
##########  DRAWING UTILITIES ###################
#################################################

def draw_3d_points(context, points, color, size):
    '''
    draw a bunch of dots
    args:
        points: a list of tuples representing x,y SCREEN coordinate eg [(10,30),(11,31),...]
        color: tuple (r,g,b,a)
        size: integer? maybe a float
    '''
    points_2d = [location_3d_to_region_2d(context.region, context.space_data.region_3d, loc) for loc in points]
    if None in points_2d:
        points_2d = filter(None, points_2d)
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


################################################
############# RAYCASTING AND SNAPPING ##########
################################################

# Jon Denning for Retopoflow
def get_ray_plane_intersection(ray_origin, ray_direction, plane_point, plane_normal):
    d = ray_direction.dot(plane_normal)
    if abs(ray_direction.dot(plane_normal)) <= 0.00000001: return float('inf')
    return (plane_point-ray_origin).dot(plane_normal) / d
# Jon Denning for Retopoflow
def get_ray_origin(ray_origin, ray_direction, ob):
    mx = ob.matrix_world
    q  = ob.rotation_quaternion
    bbox = [Vector(v) for v in ob.bound_box]
    bm = Vector((min(v.x for v in bbox),min(v.y for v in bbox),min(v.z for v in bbox)))
    bM = Vector((max(v.x for v in bbox),max(v.y for v in bbox),max(v.z for v in bbox)))
    x,y,z = Vector((1,0,0)),Vector((0,1,0)),Vector((0,0,1))
    planes = []
    if abs(ray_direction.x)>0.0001: planes += [(bm,x), (bM,-x)]
    if abs(ray_direction.y)>0.0001: planes += [(bm,y), (bM,-y)]
    if abs(ray_direction.z)>0.0001: planes += [(bm,z), (bM,-z)]
    dists = [get_ray_plane_intersection(ray_origin,ray_direction,mx*p0,q*no) for p0,no in planes]
    return ray_origin + ray_direction * min(dists)

# Jon Denning for Retopoflow
def ray_cast_region2d(region, rv3d, screen_coord, obj):
    '''
    performs ray casting on object given region, rv3d, and coords wrt region.
    returns tuple of ray vector (from coords of region) and hit info
    '''
    mx = obj.matrix_world
    rgn = region
    imx = mx.inverted()

    r2d_origin = region_2d_to_origin_3d
    r2d_vector = region_2d_to_vector_3d

    o, d = r2d_origin(rgn, rv3d, screen_coord), r2d_vector(rgn, rv3d, screen_coord).normalized()
    back = 0 if rv3d.is_perspective else 1
    mult = 100 #* (1 if rv3d.is_perspective else -1)
    bver = '%03d.%03d.%03d' % (bpy.app.version[0],bpy.app.version[1],bpy.app.version[2])
    if (bver < '002.072.000') and not rv3d.is_perspective: mult *= -1

    st, en = imx*(o-mult*back*d), imx*(o+mult*d)

    if bversion() < '002.077.000':
        hit = obj.ray_cast(st,en)
    else:
        hit = obj.ray_cast(st, en-st)
    return (d, hit)
