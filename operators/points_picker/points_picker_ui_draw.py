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
# NONE!

# Blender imports
import bpy
import blf
import bgl
from mathutils import Vector
from bpy_extras import view3d_utils

# Addon imports
from ...addon_common.cookiecutter.cookiecutter import CookieCutter
from ...functions import bgl_utils


class PointsPicker_UI_Draw():

    ###################################################
    # draw functions

    @CookieCutter.Draw("post2d")
    def draw_postpixel(self):
        context = bpy.context
        region = context.region
        rv3d = context.space_data.region_3d
        dpi = bpy.context.user_preferences.system.dpi
        if len(self.b_pts) == 0: return
        bgl_utils.draw_3d_points(context, [pt.location for pt in self.b_pts], 3)

        if self.selected != -1:
            bgl_utils.draw_3d_points(context, [self.b_pts[self.selected].location], 8, color=(0,1,1,1))

        if self.hovered[0] == 'POINT':
            bgl_utils.draw_3d_points(context, [self.b_pts[self.hovered[1]].location], 8, color=(0,1,0,1))

        # blf.size(0, 20, dpi) #fond_id = 0
        for pt in self.b_pts:
            if pt.label:
                vector2d = view3d_utils.location_3d_to_region_2d(region, rv3d, pt.location)
                blf.position(0, vector2d[0], vector2d[1], 0)
                blf.draw(0, pt.label) #font_id = 0
