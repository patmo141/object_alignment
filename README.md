# 'Object Alignment':

Iterative closest point alignment addon for Blender 
Update so that it works for Blender 3.2.2

# Instructions for Use:

* Download the latest [release](https://github.com/patmo141/object_alignment/releases) .zip file 
* In Blender go Edit > Preferences > Add-ons
* Click on Install... and select the .zip Add-on, then enable it

* Manually place the two meshes close together
* Select the base object first then the object to be aligned second and click on ICP Align
* Increase the number of ICP iterations or just ICP Align multiple times until the two meshes are well aligned

### Take m_Objects with:
This option allows to apply the same transformation matrix, when applying ICP Align to one object, to all other objects with names that start with "m_".
