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

bl_info = {
    "name"        : "Pick Points",
    "author"      : "Christopher Gearhart <chris@bblanimation.com>",
    "version"     : (1, 0, 0),
    "blender"     : (2, 80, 0),
    "description" : "",
    "location"    : "View3D > Tools > Pick Points",
    "warning"     : "",  # used for warning icon and text in addons panel
    "wiki_url"    : "",
    "tracker_url" : "",
    "category"    : "Object"}

# System imports
# NONE!

# Blender imports
import bpy
from bpy.types import Scene

# Module imports
from .operators import *
from .ui import *
from .lib import classes_to_register
from . import addon_updater_ops

def register():
    # register classes
    for cls in classes_to_register.classes:
        make_annotations(cls)
        bpy.utils.register_class(cls)

    # addon updater code and configurations
    addon_updater_ops.register(bl_info)

def unregister():
    # addon updater unregister
    addon_updater_ops.unregister()

    # unregister classes
    for cls in reversed(classes_to_register.classes):
        bpy.utils.unregister_class(cls)
