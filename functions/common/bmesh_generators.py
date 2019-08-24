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
import math

# Blender imports
import bpy
import bmesh
from mathutils import Vector

# Module imports
from .blender import select_geom
from .bmesh_utils import smooth_bm_faces


def make_rectangle(coord1:Vector, coord2:Vector, face:bool=True, flip_normal:bool=False, bme:bmesh=None):
    """
    create a rectangle with bmesh

    Keyword Arguments:
        coord1      -- back/left/bottom corner of the square (furthest negative in all three axes)
        coord2      -- front/right/top  corner of the square (furthest positive in all three axes)
        face        -- draw face connecting cube verts
        flip_normal -- flip the normals of the cube
        bme         -- bmesh object in which to create verts
    NOTE: if coord1 and coord2 are different on all three axes, z axis will stay consistent at coord1.z

    Returns:
        v_list      -- list of vertices with normal facing in positive direction (right hand rule)

    """
    # create new bmesh object
    bme = bme or bmesh.new()

    # create square with normal facing +x direction
    if coord1.x == coord2.x:
        v1, v2, v3, v4 = [bme.verts.new((coord1.x, y, z)) for y in (coord1.y, coord2.y) for z in (coord1.z, coord2.z)]
    # create square with normal facing +y direction
    elif coord1.y == coord2.y:
        v1, v2, v3, v4 = [bme.verts.new((x, coord1.y, z)) for x in (coord1.x, coord2.x) for z in (coord1.z, coord2.z)]
    # create square with normal facing +z direction
    else:
        v1, v2, v3, v4 = [bme.verts.new((x, y, coord1.z)) for x in (coord1.x, coord2.x) for y in (coord1.y, coord2.y)]
    v_list = [v1, v3, v4, v2]

    # create face
    if face:
        bme.faces.new(v_list[::-1] if flip_normal else v_list)

    return bme


def make_square(size:float, location:Vector=Vector((0, 0, 0)), face:bool=True, flip_normal:bool=False, bme:bmesh=None):
    """
    create a square with bmesh

    Keyword Arguments:
        size        -- distance of any given edge on square
        location    -- centerpoint of square
        face        -- draw face connecting cube verts
        flip_normal -- flip the normals of the cube
        bme         -- bmesh object in which to create verts

    Returns:
        v_list      -- list of vertices with normal facing in positive direction (right hand rule)

    """
    coord1 = location - Vector((size / 2, size / 2, 0))
    coord2 = location + Vector((size / 2, size / 2, 0))
    return make_rectangle(coord1, coord2, face, flip_normal, bme)


def make_cube(coord1:Vector, coord2:Vector, sides:list=[False]*6, flip_normals:bool=False, seams:bool=False, bme:bmesh=None):
    """
    create a cube with bmesh

    Keyword Arguments:
        coord1      -- back/left/bottom corner of the cube (furthest negative in all three axes)
        coord2      -- front/right/top  corner of the cube (furthest positive in all three axes)
        sides       -- draw sides [+z, -z, +x, -x, +y, -y]
        flip_normals -- flip the normals of the cube
        seams       -- make all edges seams
        bme         -- bmesh object in which to create verts

    Returns:
        v_list       -- list of vertices in the following x,y,z order: [---, -+-, ++-, +--, --+, +-+, +++, -++]

    """

    # ensure coord1 is less than coord2 in all dimensions
    assert coord1.x < coord2.x
    assert coord1.y < coord2.y
    assert coord1.z < coord2.z

    # create new bmesh object
    bme = bme or bmesh.new()

    # create vertices
    v_list = [bme.verts.new((x, y, z)) for x in (coord1.x, coord2.x) for y in (coord1.y, coord2.y) for z in (coord1.z, coord2.z)]

    # create faces
    v1, v2, v3, v4, v5, v6, v7, v8 = v_list
    new_faces = []
    if sides[0]:
        new_faces.append([v6, v8, v4, v2])
    if sides[1]:
        new_faces.append([v3, v7, v5, v1])
    if sides[4]:
        new_faces.append([v4, v8, v7, v3])
    if sides[3]:
        new_faces.append([v2, v4, v3, v1])
    if sides[2]:
        new_faces.append([v8, v6, v5, v7])
    if sides[5]:
        new_faces.append([v6, v2, v1, v5])

    for f in new_faces:
        if flip_normals:
            f.reverse()
        new_f = bme.faces.new(f)
        # if seams:
        #     for e in new_f.edges:
        #         e.seam = True

    return bme, [v1, v3, v7, v5, v2, v6, v8, v4]


def make_circle(r:float, N:int, co:tuple=Vector((0, 0, 0)), face:bool=True, flip_normals:bool=False, bme:bmesh=None):
    """
    create a circle with bmesh

    Keyword Arguments:
        r           -- radius of circle
        N           -- number of verts on circumference
        co          -- coordinate of cylinder's center
        face        -- create face between circle verts
        flip_normals -- flip normals of cylinder
        bme         -- bmesh object in which to create verts

    """
    # initialize vars
    bme = bme or bmesh.new()
    verts = []

    # create verts around circumference of circle
    for i in range(N):
        circ_val = ((2 * math.pi) / N) * (i - 0.5)
        x = r * math.cos(circ_val)
        y = r * math.sin(circ_val)
        coord = co + Vector((x, y, 0))
        verts.append(bme.verts.new(coord))
    # create face
    if face:
        bme.faces.new(verts if not flip_normals else verts[::-1])
    # create edges
    else:
        for i in range(len(verts)):
            bme.edges.new((verts[i - 1], verts[i]))

    return bme


def make_cylinder(r:float, h:float, N:int, co:Vector=Vector((0,0,0)), bot_face:bool=True, top_face:bool=True, flip_normals:bool=False, seams:bool=True, bme:bmesh=None):
    """
    create a cylinder with bmesh

    Keyword Arguments:
        r           -- radius of cylinder
        h           -- height of cylinder
        N           -- number of verts per circle
        co          -- coordinate of cylinder's center
        bot_face     -- create face on bottom of cylinder
        top_face     -- create face on top of cylinder
        flip_normals -- flip normals of cylinder
        seams       -- make horizontal edges seams
        bme         -- bmesh object in which to create verts

    """
    # initialize vars
    bme = bme or bmesh.new()
    top_verts = []
    bot_verts = []
    side_faces = []

    # create upper and lower circles
    for i in range(N):
        circ_val = ((2 * math.pi) / N) * i
        x = r * math.cos(circ_val)
        y = r * math.sin(circ_val)
        z = h / 2
        coord_t = co + Vector((x, y, z))
        coord_b = co + Vector((x, y, -z))
        top_verts.append(bme.verts.new(coord_t))
        bot_verts.append(bme.verts.new(coord_b))

    # if seams:
    #     for i in range(len(top_verts)):
    #         v1 = top_verts[i]
    #         v2 = top_verts[(i-1)]
    #         v3 = bot_verts[i]
    #         v4 = bot_verts[(i-1)]
    #         bme.edges.new((v1, v2)).seam = True
    #         bme.edges.new((v3, v4)).seam = True
    #     bme.edges.new((top_verts[0], bot_verts[0])).seam = True

    # create faces on the sides
    _, side_faces = connect_circles(top_verts if flip_normals else bot_verts, bot_verts if flip_normals else top_verts, bme)
    smooth_bm_faces(side_faces)

    # create top and bottom faces
    if top_face:
        bme.faces.new(top_verts if not flip_normals else top_verts[::-1])
    if bot_face:
        bme.faces.new(bot_verts[::-1] if not flip_normals else bot_verts)

    # return bme & dictionary with lists of top and bottom vertices
    return bme, {"bottom":bot_verts[::-1], "top":top_verts}


def make_tube(r:float, h:float, t:float, N:int, co:Vector=Vector((0,0,0)), top_face:bool=True, bot_face:bool=True, top_face_inner:bool=False, bot_face_inner:bool=False, flip_normals:bool=False, seams:bool=True, bme:bmesh=None):
    """
    create a tube with bmesh

    Keyword Arguments:
        r            -- radius of inner cylinder
        h            -- height of cylinder
        t            -- thickness of tube
        N            -- number of verts per circle
        co           -- coordinate of cylinder's center
        bot_face      -- create face on bottom of cylinder
        top_face      -- create face on top of cylinder
        bot_face_inner -- create inner circle on bottom of cylinder
        top_face_inner -- create inner circle on top of cylinder
        flip_normals  -- flip normals of cylinder
        seams       -- make horizontal edges seams
        bme          -- bmesh object in which to create verts

    """
    # create new bmesh object
    if bme == None:
        bme = bmesh.new()

    # create upper and lower circles
    bme, inner_verts = make_cylinder(r, h, N, co=co, bot_face=False, top_face=False, flip_normals=not flip_normals, bme=bme)
    bme, outer_verts = make_cylinder(r + t, h, N, co=co, bot_face=False, top_face=False, flip_normals=flip_normals, bme=bme)
    if top_face:
        connect_circles(outer_verts["top"], inner_verts["top"], bme, flip_normals=flip_normals, select=False)
    if bot_face:
        connect_circles(outer_verts["bottom"], inner_verts["bottom"], bme, flip_normals=flip_normals, select=False)
    if bot_face_inner:
        bme.faces.new(inner_verts["bottom"])
    if top_face_inner:
        bme.faces.new(inner_verts["top"][::-1])
    # return bmesh
    return bme, {"outer":outer_verts, "inner":inner_verts}


def connect_circles(circle1, circle2, bme, offset=0, flip_normals=False, smooth=True, select=True):
    assert len(circle1) - 1 > offset >= 0
    faces = []
    for v in range(len(circle1)):
        v1 = circle1[v - offset]
        v2 = circle2[v]
        v3 = circle2[(v-1)]
        v4 = circle1[(v-1) - offset]
        f = bme.faces.new([v1, v2, v3, v4][::-1 if flip_normals else 1])
        if select: select_geom((f.edges[0], f.edges[2]))
        f.smooth = smooth
        faces.append(f)
    return bme, faces
