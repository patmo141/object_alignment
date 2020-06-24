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

# Blender imports
import bpy
from bpy.types import AddonPreferences
from bpy.props import *

# updater import
from .. import addon_updater_ops


class ObjectAlignmentPreferences(AddonPreferences):
    bl_idname = __package__[:__package__.index(".lib")]

    # Addon preferences
    icp_iterations = IntProperty(
            name="ICP Iterations",
            default=50)
    redraw_frequency = IntProperty(
            name="Redraw Iterations",
            description="Number of iterations between redraw, bigger = less redraw but faster completion",
            default=10)
    use_sample = BoolProperty(
            name="Use Sample",
            description="Use a sample of verts to align",
            default=False)
    sample_fraction = FloatProperty(
            name="Sample Fraction",
            description="Only fraction of mesh verts for alignment. Less accurate, faster",
            default=0.5,
            min=0,
            max=1)
    min_start = FloatProperty(
            name="Minimum Starting Dist",
            description="Only verts closer than this distance will be used in each iteration",
            default=0.5,
            min=0,
            max=20)
    target_d = FloatProperty(
            name="Target Translation",
            description="If translation of 3 iterations is < target, ICP is considered sucessful",
            default=0.01,
            min=0,
            max=10)
    use_target = BoolProperty(
            name="Use Target",
            description="Calc alignment stats at each iteration to assess convergence. SLower per step, may result in less steps",
            default=True)
    align_methods =['RIGID','ROT_LOC_SCALE']#,'AFFINE']
    align_items = []
    for index, item in enumerate(align_methods):
        align_items.append((str(index), align_methods[index], str(index)))
    align_meth = EnumProperty(items = align_items, name="Alignment Method", description="Changes how picked points registration aligns object", default='0', options={'ANIMATABLE'}, update=None, get=None, set=None)

    
    snap_method = EnumProperty(items = [('BVH', 'BVH', 'BVH'), ('KD','KD','KD'), ('OB','OB','OB')], 
                               name="Snap Method", 
                               description="Changes paris are made", 
                               default='BVH', options={'ANIMATABLE'}, update=None, get=None, set=None)


	# addon updater preferences
    auto_check_update = BoolProperty(
        name="Auto-check for Update",
        description="If enabled, auto-check for updates using an interval",
        default=False)
    updater_intrval_months = IntProperty(
        name='Months',
        description="Number of months between checking for updates",
        default=0, min=0)
    updater_intrval_days = IntProperty(
        name='Days',
        description="Number of days between checking for updates",
        default=7, min=0)
    updater_intrval_hours = IntProperty(
        name='Hours',
        description="Number of hours between checking for updates",
        min=0, max=23,
        default=0)
    updater_intrval_minutes = IntProperty(
        name='Minutes',
        description="Number of minutes between checking for updates",
        min=0, max=59,
        default=0)

    def draw(self, context):
        layout = self.layout

        # draw addon preferences
        layout.label(text="Alignment Preferences")
        layout.prop(self, "icp_iterations")
        layout.prop(self, "redraw_frequency")
        layout.prop(self, "use_sample")
        layout.prop(self, "sample_fraction")
        layout.prop(self, "min_start")
        layout.prop(self, "use_target")
        layout.prop(self, "target_d")
        layout.prop(self, "align_meth")

        # updater draw function
        addon_updater_ops.update_settings_ui(self,context)
