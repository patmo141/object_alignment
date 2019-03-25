# 'Points Picker':

...

# Instructions for Use:

* ...

# Instructions for Use as Submodule:

* The following functions can be rewritten in a subclass:

    * `self.can_start(context)`
        * returns `True` if Points Picker ui and data structures can be initialized, else `False`
        * by default, this function checks the following, where `ob` is `bpy.context.active_object`: `return ob is not None and ob.type == "MESH"`
        * must be rewritten with the `@classmethod` decorator
    * `self.ui_setup_post()`
        * called after ui elements have been declared
        * add buttons, containers, properties, etc. to `self.tools_panel` or `self.info_panel`
        * hide existing ui elements with the following lines:
        ```
            self.info_panel.visible = False
            self.tools_panel.visible = False
            self.commit_button.visible = False
            self.cancel_button.visible = False
        ```
        * create your own ui elements
    * `self.start_post()`
        * called after ui and data structures have been initialized
    * `self.add_point_pre(loc)`
        * called before new point added at current mouse position
        * `loc` argument will be 2D Vector with new point's location
        * use to evaluate the existing points using the `self.b_pts` list or check custom conditions for adding new point
        * returns `True` if point can be added, else `False`
    * `self.add_point_post(new_point)`
        * called after new point added at current mouse position
        * `new_point` argument will D3Point object with the following attributes:
            * `new_point.label` = label string for point
            * `new_point.location` = 3D location Vector for point
            * `new_point.surface_normal` = 3D surface normal Vector of the object at this point's location
            * `new_point.view_direction` = 3D view direction Vector of the viewport at the time this point was placed
    * `self.move_point_post(moved_point)`
        * called after grabbed point has been placed
        * `moved_point` argument will D3Point object with the following attributes:
            * `new_point.label` = label string for point
            * `new_point.location` = 3D location Vector for point
            * `new_point.surface_normal` = 3D surface normal Vector of the object at this point's location
            * `new_point.view_direction` = 3D view direction Vector of the viewport at the time this point was placed
    * `self.end_commit()`
        * called when all points are committed
        * by default, this function creates new empty objects at each point location
