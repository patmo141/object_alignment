'''
Created on Apr 14, 2014

@author: Patrick
'''
bl_info = {
    "name": "Complex Alignment",
    "author": "Patrick Moore",
    "version": (0, 1),
    "blender": (2, 6, 0),
    "location": "View3D > Tools > Alignment",
    "description": "Help align objects which have overlapping featuers",
    "warning": "",
    "wiki_url": "",
    "category": "Transform Mesh"}


import numpy as np
import bpy
from bpy.types import Operator
from bpy.props import FloatVectorProperty, StringProperty, IntProperty, BoolProperty, FloatProperty
from bpy.types import Operator, AddonPreferences
from bpy_extras.object_utils import AddObjectHelper, object_data_add
from mathutils import Vector


#Preferences
class AlignmentAddonPreferences(AddonPreferences):
    # this must match the addon name, use '__package__'
    # when defining this in a submodule of a python package.
    bl_idname = __name__

    icp_iterations = IntProperty(
            name="ICP Iterations",
            default=50
            )
    
    redraw_frequency = IntProperty(
            name="Redraw Iterations",
            description = "Number of iterations between redraw, bigger = less redraw but faster completion",
            default=10)
    
    use_sampe = BoolProperty(
            name = "Use Sample",
            description = "Use a sample of verts to align",
            default = False)
    
    sample_fraction = FloatProperty(
            name="Sample Fraction",
            description = "Only fraction of mesh verts for alignment. Less accurate, faster",
            default = 0.5,
            min = 0,
            max = 1)
    
    min_start = FloatProperty(
            name="Minimum Starting Dist",
            description = "Only verts closer than this distance will be included in each iteration",
            default = 0.5,
            min = 0,
            max = 20)

    def draw(self, context):
        layout = self.layout
        layout.label(text="Alignment Preferences")
        layout.prop(self, "icp_iterations")
        layout.prop(self, "redraw_frequency")
        layout.prop(self, "use_sample")
        layout.prop(self, "sample_fraction")
        layout.prop(self, "min_start")
        

class ComplexAlignmentPanel(bpy.types.Panel):
    """UI for ICP Alignment"""
    bl_category = "Alignment"
    bl_label = "ICP Object Alignment"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'

    def draw(self, context):
        layout = self.layout

        obj = context.object

        row = layout.row()
        row.label(text="Hello world!", icon='WORLD_DATA')

        row = layout.row()
        row.label(text="Active object is: " + obj.name)
        row = layout.row()
        row.prop(obj, "name")

        row = layout.row()
        row.operator("mesh.primitive_cube_add")
#modified from http://nghiaho.com/?page_id=671    
def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.transpose(AA) * BB

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2,:] *= -1
        R = Vt.T * U.T

    t = -R*centroid_A.T + centroid_B.T

    print(t)

    return R, t





def register():
    bpy.utils.register_class(AlignmentAddonPreferences)
    bpy.utils.register_class(ComplexAlignmentPanel)


def unregister():
    bpy.utils.unregister_class(AlignmentAddonPreferences)
    bpy.utils.unregister_class(ComplexAlignmentPanel)


if __name__ == "__main__":
    register()