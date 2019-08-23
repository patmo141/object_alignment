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
        * create your own ui panels and elements
        * add/edit buttons, frames, properties, etc. in the existing structure:
        ```
            self.info_panel
                self.inst_paragraphs
            self.tools_panel
                self.commit_button
                self.cancel_button
        ```
        * hide existing ui elements with the following code (replace `self.info_panel` with any ui element above): `self.info_panel.visible = False`
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
        * called when Points Picker is committed
        * by default, this function creates new empty objects at each point location
        * must end with the following line of code: `self.end_commit_post()`
    * `self.end_commit_post()`
        * called when Points Picker is committed
    * `self.can_commit()`
        * called when the user attempts to commit Points Picker
        * by default, this function returns True
    * `self.can_cancel()`
        * called when the user attempts to cancel Points Picker
        * by default, this function returns True
