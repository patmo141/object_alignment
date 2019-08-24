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
import os
from math import *

# Blender imports
import bpy
import bmesh
from mathutils import Vector, Euler, Matrix
from bpy.types import Object, Scene
try:
    from bpy.types import ViewLayer
except ImportError:
    ViewLayer = None

# Module imports
from .python_utils import confirm_iter, confirm_list
from .wrappers import blender_version_wrapper
from .reporting import b280


#################### PREFERENCES ####################


@blender_version_wrapper("<=", "2.79")
def get_preferences(ctx=None):
    return (ctx if ctx else bpy.context).user_preferences
@blender_version_wrapper(">=", "2.80")
def get_preferences(ctx=None):
    return (ctx if ctx else bpy.context).preferences


def get_addon_preferences():
    """ get preferences for current addon """
    if not hasattr(get_addon_preferences, "prefs"):
        folderpath, foldername = os.path.split(get_addon_directory())
        addons = get_preferences().addons
        if not addons[foldername].preferences:
            return None
        get_addon_preferences.prefs = addons[foldername].preferences
    return get_addon_preferences.prefs


def get_addon_directory():
    """ get root directory of current addon """
    addons = get_preferences().addons
    folderpath = os.path.dirname(os.path.abspath(__file__))
    while folderpath:
        folderpath, foldername = os.path.split(folderpath)
        if foldername in {"common", "functions", "addons"}:
            continue
        if foldername in addons:
            break
    else:
        raise NameError("Did not find addon directory")
    return os.path.join(folderpath, foldername)


#################### OBJECTS ####################


def delete(objs, remove_meshes:bool=False):
    """ efficient deletion of objects """
    objs = confirm_iter(objs)
    for obj in objs:
        if obj is None:
            continue
        if remove_meshes:
            m = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if remove_meshes and m is not None:
            bpy.data.meshes.remove(m)


def duplicate(obj:Object, linked:bool=False, link_to_scene:bool=False):
    """ efficient duplication of objects """
    copy = obj.copy()
    if not linked and copy.data:
        copy.data = copy.data.copy()
    unhide(copy, render=False)
    if link_to_scene:
        link_object(copy)
    return copy


@blender_version_wrapper("<=","2.79")
def set_active_obj(obj:Object, scene:Scene=None):
    scene = scene or bpy.context.scene
    scene.objects.active = obj
@blender_version_wrapper(">=","2.80")
def set_active_obj(obj:Object, view_layer:ViewLayer=None):
    view_layer = view_layer or bpy.context.view_layer
    view_layer.objects.active = obj


@blender_version_wrapper("<=","2.79")
def select(obj_list, active:bool=False, only:bool=False):
    """ selects objs in list (deselects the rest if 'only') """
    # confirm obj_list is a list of objects
    obj_list = confirm_iter(obj_list)
    # deselect all if selection is exclusive
    if only:
        deselect_all()
    # select objects in list
    for obj in obj_list:
        if obj is not None and not obj.select:
            obj.select = True
    # set active object
    if active:
        set_active_obj(obj_list[0])
@blender_version_wrapper(">=","2.80")
def select(obj_list, active:bool=False, only:bool=False):
    """ selects objs in list (deselects the rest if 'only') """
    # confirm obj_list is a list of objects
    obj_list = confirm_iter(obj_list)
    # deselect all if selection is exclusive
    if only:
        deselect_all()
    # select objects in list
    for obj in obj_list:
        if obj is not None and not obj.select_get():
            obj.select_set(True)
    # set active object
    if active:
        set_active_obj(obj_list[0])


def select_all():
    """ selects all objs in scene """
    select(bpy.context.scene.objects)


def select_geom(geom, only:bool=False):
    """ selects verts/edges/faces in list and deselects the rest """
    # confirm vertList is a list of vertices
    geom = confirm_list(geom)
    # deselect all if selection is exclusive
    if only: deselect_all()
    # select vertices in list
    for v in geom:
        if v is not None and not v.select:
            v.select = True


@blender_version_wrapper("<=","2.79")
def deselect(obj_list):
    """ deselects objs in list """
    # confirm obj_list is a list of objects
    obj_list = confirm_list(obj_list)
    # select/deselect objects in list
    for obj in obj_list:
        if obj is not None and obj.select:
            obj.select = False
@blender_version_wrapper(">=","2.80")
def deselect(obj_list):
    """ deselects objs in list """
    # confirm obj_list is a list of objects
    obj_list = confirm_list(obj_list)
    # select/deselect objects in list
    for obj in obj_list:
        if obj is not None and obj.select_get():
            obj.select_set(False)


@blender_version_wrapper("<=","2.79")
def deselect_all():
    """ deselects all objs in scene """
    for obj in bpy.context.selected_objects:
        if obj.select:
            obj.select = False
@blender_version_wrapper(">=","2.80")
def deselect_all():
    """ deselects all objs in scene """
    selected_objects = bpy.context.selected_objects if hasattr(bpy.context, "selected_objects") else [obj for obj in bpy.context.view_layer.objects if obj.select_get()]
    deselect(selected_objects)


@blender_version_wrapper("<=","2.79")
def is_selected(obj):
    return obj.select
@blender_version_wrapper(">=","2.80")
def is_selected(obj):
    return obj.select_get()


@blender_version_wrapper("<=","2.79")
def hide(obj:Object, viewport:bool=True, render:bool=True):
    if not obj.hide and viewport:
        obj.hide = True
    if not obj.hide_render and render:
        obj.hide_render = True
@blender_version_wrapper(">=","2.80")
def hide(obj:Object, viewport:bool=True, render:bool=True):
    if not obj.hide_viewport and viewport:
        obj.hide_viewport = True
    if not obj.hide_render and render:
        obj.hide_render = True


@blender_version_wrapper("<=","2.79")
def unhide(obj:Object, viewport:bool=True, render:bool=True):
    if obj.hide and viewport:
        obj.hide = False
    if obj.hide_render and render:
        obj.hide_render = False
@blender_version_wrapper(">=","2.80")
def unhide(obj:Object, viewport:bool=True, render:bool=True):
    if obj.hide_viewport and viewport:
        obj.hide_viewport = False
    if obj.hide_render and render:
        obj.hide_render = False


@blender_version_wrapper("<=","2.79")
def is_obj_visible_in_viewport(obj:Object):
    scn = bpy.context.scene
    return any([obj.layers[i] and scn.layers[i] for i in range(20)])
@blender_version_wrapper(">=","2.80")
def is_obj_visible_in_viewport(obj:Object):
    if obj is None:
        return False
    obj_visible = not obj.hide_viewport
    if obj_visible:
        for cn in obj.users_collection:
            if cn.hide_viewport:
                obj_visible = False
                break
    return obj_visible


@blender_version_wrapper("<=","2.79")
def link_object(o:Object, scene:Scene=None):
    scene = scene or bpy.context.scene
    scene.objects.link(o)
@blender_version_wrapper(">=","2.80")
def link_object(o:Object, scene:Scene=None):
    scene = scene or bpy.context.scene
    scene.collection.objects.link(o)


@blender_version_wrapper("<=","2.79")
def unlink_object(o:Object):
    bpy.context.scene.objects.unlink(o)
@blender_version_wrapper(">=","2.80")
def unlink_object(o:Object):
    for coll in o.users_collection:
        coll.objects.unlink(o)


@blender_version_wrapper("<=","2.79")
def safe_link(obj:Object, protect:bool=False, collections=None):
    # link object to scene
    try:
        link_object(obj)
    except RuntimeError:
        pass
    # remove fake user from object data
    obj.use_fake_user = False
    # protect object from deletion (useful in Bricker addon)
    if hasattr(obj, "protected"):
        obj.protected = protect
@blender_version_wrapper(">=","2.80")
def safe_link(obj:Object, protect:bool=False, collections=None):
    # link object to target collections (scene collection by default)
    collections = collections or [bpy.context.scene.collection]
    for coll in collections:
        try:
            coll.objects.link(obj)
        except RuntimeError:
            continue
    # remove fake user from object data
    obj.use_fake_user = False
    # protect object from deletion (useful in Bricker addon)
    if hasattr(obj, "protected"):
        obj.protected = protect


def safe_unlink(obj:Object, protect:bool=True):
    # unlink object from scene
    try:
        unlink_object(obj)
    except RuntimeError:
        pass
    # prevent object data from being tossed on Blender exit
    obj.use_fake_user = True
    # protect object from deletion (useful in Bricker addon)
    if hasattr(obj, "protected"):
        obj.protected = protect


def copy_animation_data(source:Object, target:Object):
    """ copy animation data from one object to another """
    if source.animation_data is None:
        return

    ad = source.animation_data

    properties = [p.identifier for p in ad.bl_rna.properties if not p.is_readonly]

    if target.animation_data is None:
        target.animation_data_create()
    ad2 = target.animation_data

    for prop in properties:
        setattr(ad2, prop, getattr(ad, prop))


def insert_keyframes(objs, keyframeType:str, frame:int, if_needed:bool=False):
    """ insert key frames for given objects to given frames """
    objs = confirm_iter(objs)
    options = set(["INSERTKEY_NEEDED"] if if_needed else [])
    for obj in objs:
        inserted = obj.keyframe_insert(data_path=keyframe_type, frame=frame, options=options)


@blender_version_wrapper("<=", "2.79")
def new_mesh_from_object(obj:Object):
    return bpy.data.meshes.new_from_object(bpy.context.scene, obj, apply_modifiers=True, settings="PREVIEW")
@blender_version_wrapper(">=", "2.80")
def new_mesh_from_object(obj:Object):
    depsgraph = bpy.context.view_layer.depsgraph
    obj_eval = obj.evaluated_get(depsgraph)
    return bpy.data.meshes.new_from_object(obj_eval)


def apply_modifiers(obj:Object):
    """ apply modifiers to object """
    m = new_mesh_from_object(obj)
    obj.modifiers.clear()
    obj.data = m


@blender_version_wrapper("<=","2.79")
def light_add(type:str="POINT", radius:float=1.0, align:str="WORLD", location:tuple=(0.0, 0.0, 0.0), rotation:tuple=(0.0, 0.0, 0.0)):
    view_align = align != "WORLD"
    bpy.ops.object.lamp_add(type=type, radius=radius, view_align=view_align, location=location, rotation=rotation)
@blender_version_wrapper(">=","2.80")
def light_add(type:str="POINT", radius:float=1.0, align:str="WORLD", location:tuple=(0.0, 0.0, 0.0), rotation:tuple=(0.0, 0.0, 0.0)):
    bpy.ops.object.light_add(type=type, radius=radius, align=align, location=location, rotation=rotation)


def is_smoke(ob:Object):
    """ check if object is smoke domain """
    if ob is None:
        return False
    for mod in ob.modifiers:
        if mod.type == "SMOKE" and mod.domain_settings and mod.show_viewport:
            return True
    return False


def is_adaptive(ob:Object):
    """ check if smoke domain object uses adaptive domain """
    if ob is None:
        return False
    for mod in ob.modifiers:
        if mod.type == "SMOKE" and mod.domain_settings and mod.domain_settings.use_adaptive_domain:
            return True
    return False


#################### VIEWPORT ####################


def tag_redraw_areas(area_types:iter=["ALL"]):
    """ run tag_redraw for given area types """
    area_types = confirm_list(area_types)
    screens = [bpy.context.screen] if bpy.context.screen else bpy.data.screens
    for screen in screens:
        for area in screen.areas:
            for areaType in area_types:
                if areaType == "ALL" or area.type == areaType:
                    area.tag_redraw()


@blender_version_wrapper("<=", "2.79")
def disable_relationship_lines():
    """ disable relationship lines in VIEW_3D """
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            area.spaces[0].show_relationship_lines = False
@blender_version_wrapper(">=", "2.80")
def disable_relationship_lines():
    """ disable relationship lines in VIEW_3D """
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            area.spaces[0].overlay.show_relationship_lines = False


def set_active_scene(scn:Scene):
    """ set active scene in all screens """
    for screen in bpy.data.screens:
        screen.scene = scn


def change_context(context, areaType:str):
    """ Changes current context and returns previous area type """
    last_area_type = context.area.type
    context.area.type = areaType
    return last_area_type


def assemble_override_context(area_type="VIEW_3D"):
    """
    Iterates through the blender GUI's areas & regions to find the View3D space
    NOTE: context override can only be used with bpy.ops that were called from a window/screen with a view3d space
    """
    win      = bpy.context.window
    scr      = win.screen
    areas3d  = [area for area in scr.areas if area.type == area_type]
    region   = [region for region in areas3d[0].regions if region.type == "WINDOW"]
    override = {"window": win,
                "screen": scr,
                "area"  : areas3d[0],
                "region": region[0],
                "scene" : bpy.context.scene,
                }
    return override


@blender_version_wrapper("<=","2.79")
def set_layers(layers:iter, scn:Scene=None):
    """ set active layers of scn w/o 'dag ZERO' error """
    assert len(layers) == 20
    scn = scn or bpy.context.scene
    # update scene (prevents dag ZERO errors)
    scn.update()
    # set active layers of scn
    scn.layers = layers


@blender_version_wrapper("<=","2.79")
def open_layer(layer_num:int, scn:Scene=None):
    scn = scn or bpy.context.scene
    layer_list = [i == layer_num - 1 for i in range(20)]
    scn.layers = layer_list
    return layer_list


#################### MESHES ####################


def draw_bmesh(bm:bmesh, name:str="drawn_bmesh"):
    """ create mesh and object from bmesh """
    # note: neither are linked to the scene, yet, so they won't show in the 3d view
    m = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, m)

    link_object(obj)          # link new object to scene
    select(obj, active=True)  # select new object and make active (does not deselect other objects)
    bm.to_mesh(m)             # push bmesh data into m
    return obj


def smooth_mesh_faces(faces:iter):
    """ set given Mesh faces to smooth """
    faces = confirm_iter(faces)
    for f in faces:
        f.use_smooth = True


#################### OTHER ####################


@blender_version_wrapper("<=","2.79")
def active_render_engine():
    return bpy.context.scene.render.engine
@blender_version_wrapper(">=","2.80")
def active_render_engine():
    return bpy.context.engine


@blender_version_wrapper("<=","2.79")
def update_depsgraph():
    bpy.context.scene.update()
@blender_version_wrapper(">=","2.80")
def update_depsgraph():
    bpy.context.view_layer.depsgraph.update()


@blender_version_wrapper("<=","2.79")
def right_align(layout_item):
    pass
@blender_version_wrapper(">=","2.80")
def right_align(layout_item):
    layout_item.use_property_split = True
    layout_item.use_property_decorate = False


def get_item_by_id(collection, id:int):
    """ get UIlist item from collection with given id """
    success = False
    for item in collection:
        if item.id == id:
            success = True
            break
    return item if success else None


@blender_version_wrapper("<=","2.79")
def layout_split(layout, align=True, factor=0.5):
    return layout.split(align=align, percentage=factor)
@blender_version_wrapper(">=","2.80")
def layout_split(layout, align=True, factor=0.5):
    return layout.split(align=align, factor=factor)


@blender_version_wrapper("<=","2.79")
def bpy_collections():
    return bpy.data.groups
@blender_version_wrapper(">=","2.80")
def bpy_collections():
    return bpy.data.collections


@blender_version_wrapper("<=","2.79")
def set_active_scene(scene:Scene):
    bpy.context.screen.scene = scene
@blender_version_wrapper(">=","2.80")
def set_active_scene(scene:Scene):
    bpy.context.window.scene = scene


def set_cursor(cursor):
    # DEFAULT, NONE, WAIT, CROSSHAIR, MOVE_X, MOVE_Y, KNIFE, TEXT,
    # PAINT_BRUSH, HAND, SCROLL_X, SCROLL_Y, SCROLL_XY, EYEDROPPER
    for wm in bpy.data.window_managers:
        for win in wm.windows:
            win.cursor_modal_set(cursor)


@blender_version_wrapper("<=","2.79")
def get_cursor_location():
    return bpy.context.scene.cursor_location
@blender_version_wrapper(">=","2.80")
def get_cursor_location():
    return bpy.context.scene.cursor.location


@blender_version_wrapper("<=","2.79")
def set_cursor_location(loc:tuple):
    bpy.context.scene.cursor_location = loc
@blender_version_wrapper(">=","2.80")
def set_cursor_location(loc:tuple):
    bpy.context.scene.cursor.location = loc


@blender_version_wrapper("<=","2.79")
def make_annotations(cls):
    """Does nothing in Blender 2.79"""
    return cls
@blender_version_wrapper(">=","2.80")
def make_annotations(cls):
    """Converts class fields to annotations in Blender 2.8"""
    bl_props = {k: v for k, v in cls.__dict__.items() if isinstance(v, tuple)}
    if bl_props:
        if "__annotations__" not in cls.__dict__:
            setattr(cls, "__annotations__", {})
        annotations = cls.__dict__["__annotations__"]
        for k, v in bl_props.items():
            annotations[k] = v
            delattr(cls, k)
    return cls


@blender_version_wrapper("<=","2.79")
def get_annotations(cls):
    return list(dict(cls).keys())
@blender_version_wrapper(">=","2.80")
def get_annotations(cls):
    return cls.__annotations__


def get_attr_folder(data_attr):
    if data_attr == "brushes":
        attr_folder = "Brush"
    elif data_attr == "meshes":
        attr_folder = "Mesh"
    elif data_attr == "libraries":
        attr_folder = "Library"
    elif data_attr == "metaballs":
        attr_folder = "MetaBall"
    elif data_attr == "movieclips":
        attr_folder = "MovieClip"
    elif data_attr == "workspace":
        attr_folder = "WorkSpace"
    else:
        attr_folder = data_attr.title().replace("_", "")[:-1]
    assert hasattr(bpy.types, attr_folder)
    return attr_folder


def append_from(blendfile_path, data_attr, filename):
    attr_folder = get_attr_folder(data_attr)
    directory = os.path.join(blendfile_path, attr_folder)
    filepath = os.path.join(directory, filename)
    bpy.ops.wm.append(
        filepath=filepath,
        filename=filename,
        directory=directory)


def append_all_from(blendfile_path, data_attr, overwrite_data=False):
    data_block_infos = list()
    orig_data_names = lambda: None
    with bpy.data.libraries.load(blendfile_path) as (data_from, data_to):
        setattr(data_to, data_attr, getattr(data_from, data_attr))
        # store copies of loaded attributes to 'orig_data_names' object
        if overwrite_data:
            attrib = getattr(data_from, data_attr)
            if len(attrib) > 0:
                setattr(orig_data_names, data_attr, attrib.copy())
    # overwrite existing data with loaded data of the same name
    if overwrite_data:
        # get attributes to remap
        source_attr = getattr(orig_data_names, data_attr)
        target_attr = getattr(data_to, data_attr)
        for i, data_name in enumerate(source_attr):
            # check that the data doesn't match
            if not hasattr(target_attr[i], "name") or target_attr[i].name == data_name or not hasattr(bpy.data, data_attr): continue
            # remap existing data to loaded data
            data_group = getattr(bpy.data, data_attr)
            data_group.get(data_name).user_remap(target_attr[i])
            # remove remapped existing data
            data_group.remove(data_group.get(data_name))
            # rename loaded data to original name
            target_attr[i].name = data_name
    return data_to
