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
# NONE!

# Addon imports
from ...addon_common.common import ui


class Skeleton_UI_Init():
    def ui_setup(self):
        # instructions
        self.instructions = {
            "do something": "Do something...",
            "do something else": "Do something else...",
        }

        # UI Box functionality
        def get_blobsize(): return self.skeleton_opts["size"]
        def get_blobsize_print(): return "%0.3f" % self.skeleton_opts["size"]
        def set_blobsize(v): self.skeleton_opts["size"] = min(max(0.001, float(v)),8.0)

        def get_action(): return self.skeleton_opts["action"]
        def set_action(v): self.skeleton_opts["action"] = v

        def mode_getter(): return self._state
        def mode_setter(m): self.fsm_change(m)

        # UPPER LEFT WINDOW, mode setters and commit/cancel buttons
        self.tools_panel = self.wm.create_window('Skeleton Tools', {'pos':7, 'movable':True, 'bgcolor':(0.50, 0.50, 0.50, 0.90)})
        precut_container = self.tools_panel.add(ui.UI_Container()) # TODO: make this rounded
        self.mode_frame = precut_container.add(ui.UI_Frame('Skeleton Mode'))
        self.mode_options = self.mode_frame.add(ui.UI_Options(mode_getter, mode_setter, separation=0))
        self.mode_options.add_option('Some State', value='some_state')

        segmentation_container = self.tools_panel.add(ui.UI_Container())
        self.finish_frame = segmentation_container.add(ui.UI_Frame('Finish Tools'))
        self.commit_button = self.finish_frame.add(ui.UI_Button('Commit', self.done, align=0))
        self.cancel_button = self.finish_frame.add(ui.UI_Button('Cancel', lambda:self.done(cancel=True), align=0))

        #####################################
        ### Collapsible Help and Options   ##
        #####################################
        self.info_panel = self.wm.create_window('Skeleton Help',
                                                {'pos':9,
                                                 'movable':True,
                                                 'bgcolor':(0.50, 0.50, 0.50, 0.90)})

        collapse_container = self.info_panel.add(ui.UI_Collapsible('Instructions     ', collapsed=False))
        self.inst_paragraphs = [collapse_container.add(ui.UI_Markdown('', min_size=(100,10), max_size=(250, 20))) for i in range(2)]
        self.set_ui_text()
        #for i in self.inst_paragraphs: i.visible = False
        self.options_frame = self.info_panel.add(ui.UI_Frame('Tool Options'))
        self.options_frame.add(ui.UI_Number("Size", get_blobsize, set_blobsize, fn_get_print_value=get_blobsize_print, fn_set_print_value=set_blobsize))
        self.wax_action_options = self.options_frame.add(ui.UI_Options(get_action, set_action, label="Action: ", vertical=True))
        self.wax_action_options.add_option("do something")
        self.wax_action_options.add_option("do something else")
        self.wax_action_options.add_option("none")

    def set_ui_text(self):
        ''' sets the viewports text '''
        self.reset_ui_text()
        for i,val in enumerate(['do something', 'do something else']):
            self.inst_paragraphs[i].set_markdown("- " + self.instructions[val])

    def reset_ui_text(self):
        for inst_p in self.inst_paragraphs:
            inst_p.set_markdown('')
