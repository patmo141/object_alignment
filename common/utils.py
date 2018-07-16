'''
Copyright (C) 2017 CG Cookie
http://cgcookie.com
hello@cgcookie.com

Created by Jonathan Denning, Jonathan Williamson

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
'''

import os

import bpy
from bmesh.types import BMesh, BMVert, BMEdge, BMFace
from mathutils import Vector, Matrix

from .profiler import profiler
from .debug import dprint
from .maths import (
    Point, Direction, Normal, Frame,
    Point2D, Vec2D, Direction2D,
    Ray, XForm, BBox, Plane
)


StructRNA = bpy.types.bpy_struct
def still_registered(self, oplist):
    if getattr(still_registered, 'is_broken', False): return False
    def is_registered():
        cur = bpy.ops
        for n in oplist:
            if not hasattr(cur, n): return False
            cur = getattr(cur, n)
        try:    StructRNA.path_resolve(self, "properties")
        except:
            print('no properties!')
            return False
        return True
    if is_registered(): return True
    still_registered.is_broken = True
    print('bpy.ops.%s is no longer registered!' % '.'.join(oplist))
    return False

registered_objects = {}
def registered_object_add(self):
    global registered_objects
    opid = self.operator_id
    print('Registering bpy.ops.%s' % opid)
    registered_objects[opid] = (self, opid.split('.'))

def registered_check():
    global registered_objects
    return all(still_registered(s, o) for (s, o) in registered_objects.values())

def selection_mouse():
    select_type = bpy.context.user_preferences.inputs.select_mouse
    return ['%sMOUSE' % select_type, 'SHIFT+%sMOUSE' % select_type]

def get_settings():
    if not hasattr(get_settings, 'cache'):
        addons = bpy.context.user_preferences.addons
        folderpath = os.path.dirname(os.path.abspath(__file__))
        while folderpath:
            folderpath,foldername = os.path.split(folderpath)
            if foldername in {'lib','addons'}: continue
            if foldername in addons: break
        else:
            assert False, 'Could not find non-"lib" folder'
        if not addons[foldername].preferences: return None
        get_settings.cache = addons[foldername].preferences
    return get_settings.cache

def get_dpi():
    system_preferences = bpy.context.user_preferences.system
    factor = getattr(system_preferences, "pixel_size", 1)
    return int(system_preferences.dpi * factor)

def get_dpi_factor():
    return get_dpi() / 72

def blender_version():
    major,minor,rev = bpy.app.version
    # '%03d.%03d.%03d' % (major, minor, rev)
    return '%d.%02d' % (major,minor)


def iter_running_sum(lw):
    s = 0
    for w in lw:
        s += w
        yield (w,s)

def iter_pairs(items, wrap, repeat=False):
    if not items: return
    while True:
        for i0,i1 in zip(items[:-1],items[1:]): yield i0,i1
        if wrap: yield items[-1],items[0]
        if not repeat: return

def rotate_cycle(cycle, offset):
    l = len(cycle)
    return [cycle[(l + ((i - offset) % l)) % l] for i in range(l)]

def max_index(vals, key=None):
    if not key: return max(enumerate(vals), key=lambda ival:ival[1])[0]
    return max(enumerate(vals), key=lambda ival:key(ival[1]))[0]

def min_index(vals, key=None):
    if not key: return min(enumerate(vals), key=lambda ival:ival[1])[0]
    return min(enumerate(vals), key=lambda ival:key(ival[1]))[0]


def shorten_floats(s):
    # reduces number of digits (for float) found in a string
    # useful for reducing noise of printing out a Vector, Buffer, Matrix, etc.
    s = re.sub(r'(?P<neg>-?)(?P<d0>\d)\.(?P<d1>\d)\d\d+e-02', r'\g<neg>0.0\g<d0>\g<d1>', s)
    s = re.sub(r'(?P<neg>-?)(?P<d0>\d)\.\d\d\d+e-03', r'\g<neg>0.00\g<d0>', s)
    s = re.sub(r'-?\d\.\d\d\d+e-0[4-9]', r'0.000', s)
    s = re.sub(r'-?\d\.\d\d\d+e-[1-9]\d', r'0.000', s)
    s = re.sub(r'(?P<digs>\d\.\d\d\d)\d+', r'\g<digs>', s)
    return s




class UniqueCounter():
    __counter = 0
    @staticmethod
    def next():
        UniqueCounter.__counter += 1
        return UniqueCounter.__counter
