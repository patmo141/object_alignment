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
from ...addon_common.cookiecutter.cookiecutter import CookieCutter
from ...addon_common.common import ui


class Skeleton_States():

    #############################################
    # State keymap

    default_keymap = {
        "switch_state": {"SHIFT+C","CTRL+LEFTMOUSE"},
        "commit": {"RET"},
        "cancel": {"ESC"},
    }

    #############################################
    # State functions

    #--------------------------------------
    # main

    @CookieCutter.FSM_State("main")
    def modal_main(self):
        # switch state
        if self.actions.pressed("switch_state"):
            return "some_state"
        # other actions
        if self.actions.pressed("commit"):
            self.done();
            return
        if self.actions.pressed("cancel"):
            self.done(cancel=True)
            return

    #--------------------------------------
    # another state

    @CookieCutter.FSM_State("some_state", "can enter")
    def can_enter_some_state(self):
        return True

    @CookieCutter.FSM_State("some_state", "enter")
    def enter_some_state(self):
        pass

    @CookieCutter.FSM_State("some_state")
    def modal_some_state(self):
        # some_state actions
        if self.skeleton_opts["action"] == "do something":
            self.do_something()
        elif self.skeleton_opts["action"] == "do something else":
            self.do_something_else()
        # other actions
        if self.actions.pressed("commit"):
            self.done();
            return
        if self.actions.pressed("cancel"):
            self.done(cancel=True)
            return

    @CookieCutter.FSM_State("some_state", "can exit")
    def can_exit_some_state(self):
        return True

    @CookieCutter.FSM_State("some_state", "exit")
    def exit_some_state(self):
        pass

    #############################################
