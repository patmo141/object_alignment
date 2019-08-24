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
# NONE!

# Blender imports
import bpy
from bpy.types import Panel

# Module imports
from ..functions import *


class VIEW3D_PT_object_alignment(Panel):
    """UI for Object Alignment Addon"""
    bl_space_type  = "VIEW_3D"
    bl_region_type = "UI" if b280() else "TOOLS"
    bl_label       = "Object Alignment"
    bl_context     = "objectmode"
    bl_category    = "Object Alignment"

    def draw(self, context):
        settings = get_addon_preferences()
        layout = self.layout

        if bpy.data.texts.find("Object Alignment log") >= 0:
            split = layout.split(align=True, percentage=0.9)
            col = split.column(align=True)
            row = col.row(align=True)
            row.operator("scene.report_error", text="Report Error", icon="URL")
            col = split.column(align=True)
            row = col.row(align=True)
            row.operator("scene.close_report_error", text="", icon="PANEL_CLOSE")

        row = layout.row()
        row.label(text="Alignment Tools", icon="MOD_SHRINKWRAP")

        align_obj = context.object
        if align_obj:
            row = layout.row()
            row.label(text="Align object is: " + align_obj.name)
        else:
            row.label(text="No Alignment Object!")

        if len(context.selected_objects) == 2:
            base_obj = [obj for obj in context.selected_objects if obj != align_obj][0]
            row = layout.row()
            row.label(text="Base object is: " + base_obj.name)
        else:
            row = layout.row()
            row.label(text="No Base object!")

        row = layout.row()
        row.label(text="Pre Processing")
        row = layout.row()
        row.operator("object.align_include")
        row.operator("object.align_include_clear", icon="X", text="")

        row = layout.row()
        row.operator("object.align_exclude")
        row.operator("object.align_exclude_clear", icon="X", text="")

        row = layout.row()
        row.label(text="Initial Alignment")
        row = layout.row()
        row.operator("object.align_picked_points")
        row.operator("screen.area_dupli", icon="FULLSCREEN_ENTER", text="")

        row = layout.row()
        row.prop(settings, "align_meth")

        row = layout.row()
        row.label(text="Iterative Alignment")
        row = layout.row()
        row.operator("object.align_icp")

        row = layout.row()
        row.operator("object.align_icp_redraw")
        row = layout.row()
        row.prop(settings, "redraw_frequency")
        row = layout.row()
        row.prop(settings, "icp_iterations")
        row = layout.row()
        row.prop(settings, "use_sample")
        row.prop(settings, "sample_fraction")
        row = layout.row()
        row.prop(settings, "min_start")
        row = layout.row()
        row.prop(settings, "use_target")
        row.prop(settings, "target_d")
