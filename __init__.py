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

bl_info = {
    "name": "Object Alignment",
    "author": "Patrick Moore, Christopher Gearhart & Alexandre Cavaleri",
    "version": (0, 3, 0),
    "blender": (3, 2, 2),
    "location": "View3D > UI > Alignment",
    "description": "Help align objects which have overlapping features",
    "warning": "",
    "wiki_url": "",
    "category": "Transform Mesh"
}

# System imports
# NONE!

# Blender imports
import bpy

# Addon imports
from .operators import *
from .ui import *
from .lib.classesToRegister import classes
from .functions.common import *
from . import addon_updater_ops

def register():
    # register classes
    for cls in classes:
        make_annotations(cls)
        bpy.utils.register_class(cls)

    # # register app handlers
    # bpy.app.handlers.load_post.append(handle_something)

    # addon updater code and configurations
    addon_updater_ops.register(bl_info)

def unregister():
    # addon updater unregister
    addon_updater_ops.unregister()

    # # unregister app handlers
    # bpy.app.handlers.load_post.remove(handle_something)

    # unregister classes
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
