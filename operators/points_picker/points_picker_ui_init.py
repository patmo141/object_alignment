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
import time
import random

# Blender imports
from bpy_extras import view3d_utils

# Addon imports
from ...addon_common.cookiecutter.cookiecutter import CookieCutter
from ...addon_common.common import ui
from ...addon_common.common.blender import show_error_message
from ...addon_common.common.ui import Drawing


class PointsPicker_UI_Init():

    ###################################################
    # draw init

    def ui_setup(self):
        # UI Box functionality
        # def get_blobsize(): return self.wax_opts["blob_size"]
        # def get_blobsize_print(): return "%0.3f" % self.wax_opts["blob_size"]
        # def set_blobsize(v): self.wax_opts["blob_size"] = min(max(0.001, float(v)),8.0)

        # instructions
        self.instructions = {
            "add": "Press left-click to add or select a point",
            "grab": "Hold left-click on a point and drag to move it along the surface of the mesh",
            "remove": "Press 'ALT' and left-click to remove a point",
        }

        def mode_getter(): return self._state
        def mode_setter(m): self.fsm_change(m)

        win_tools = self.wm.create_window('Points Picker Tools', {'pos':7, 'movable':True, 'bgcolor':(0.50, 0.50, 0.50, 0.90)})

        precut_container = win_tools.add(ui.UI_Container()) # TODO: make this rounded

        # container = precut_container.add(ui.UI_Frame('Points Picker Mode'))
        # wax_mode = container.add(ui.UI_Options(mode_getter, mode_setter, separation=0))
        # wax_mode.add_option('Sketch', value='sketch wait')
        # wax_mode.add_option('Paint', value='paint wait')

        segmentation_container = win_tools.add(ui.UI_Container())
        container = segmentation_container.add(ui.UI_Frame('Points Picker Tools'))
        container.add(ui.UI_Button('Commit', self.done, align=0))
        container.add(ui.UI_Button('Cancel', lambda:self.done(cancel=True), align=0))

        info = self.wm.create_window('Points Picker Help', {'pos':9, 'movable':True})#, 'bgcolor':(0.30, 0.60, 0.30, 0.90)})
        info.add(ui.UI_Label('Instructions', align=0, margin=4))
        self.inst_paragraphs = [info.add(ui.UI_Markdown('', min_size=(200,10))) for i in range(3)]
        #for i in self.inst_paragraphs: i.visible = False
        #self.ui_instructions = info.add(ui.UI_Markdown('test', min_size=(200,200)))
        # opts = info.add(ui.UI_Frame('Tool Options'))
        # opts.add(ui.UI_Number("Size", get_blobsize, set_blobsize, fn_get_print_value=get_blobsize_print, fn_set_print_value=set_blobsize))

        self.set_ui_text()

    # XXX: Fine for now, but will likely be irrelevant in future
    def set_ui_text(self):
        ''' sets the viewports text '''
        self.reset_ui_text()
        for i,val in enumerate(['add', 'grab', 'remove']):
            self.inst_paragraphs[i].set_markdown(chr(65 + i) + ") " + self.instructions[val])

    def reset_ui_text(self):
        for inst_p in self.inst_paragraphs:
            inst_p.set_markdown('')

    ###################################################
