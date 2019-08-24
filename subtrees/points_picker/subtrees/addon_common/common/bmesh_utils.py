'''
Copyright (C) 2018 CG Cookie
http://cgcookie.com
hello@cgcookie.com

Created by Jonathan Denning and Jonathan Williamson

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


class BMeshState:
    def __init__(self, bmesh, property, default_value=False):
        self.bmesh = bmesh
        self.property = property
        self.default_value = default_value
        self.vert_state = { bmv:getattr(bmv, property) for bmv in bmesh.verts }
        self.edge_state = { bme:getattr(bme, property) for bme in bmesh.edges }
        self.face_state = { bmf:getattr(bmf, property) for bmf in bmesh.faces }

    def restore(self, verts=True, edges=True, faces=True):
        if faces:
            for bmf in self.bmesh.faces:
                setattr(bmf, self.property, self.face_state.get(bmf, self.default_value))
        if edges:
            for bme in self.bmesh.edges:
                setattr(bme, self.property, self.edge_state.get(bme, self.default_value))
        if verts:
            for bmv in self.bmesh.verts:
                setattr(bmv, self.property, self.vert_state.get(bmv, self.default_value))


class BMeshSelectState(BMeshState):
    ''' Saves selection state of BMesh, allowing to restore '''
    def __init__(self, bmesh):
        super().__init__(bmesh, 'select')


class BMeshHideState(BMeshState):
    ''' Saves hide state of BMesh, allowing to restore '''
    def __init__(self, bmesh):
        super().__init__(bmesh, 'hide')


