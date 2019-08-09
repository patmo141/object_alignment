# addon_common

This repo contains the CookieCutter Blender add-on framework.

The main branch works only with Blender 2.79, but development for Blender 2.80 is under the `b280` branch.


## Example Add-on

As an example add-on for Blender 2.79, see the [ExtruCut](https://github.com/CGCookie/ExtruCut) project.


## Creating your own add-on using CookieCutter

### Creating a Blender 2.79b add-on

```
# create new addon folder
mkdir newaddon
cd newaddon

# initialize as new git repo
git init .

# add CC addon_common as submodule
git submodule add git@github.com:CGCookie/addon_common.git addon_common
```

### Creating a Blender 2.80 add-on

```
# create new addon folder
mkdir newaddon
cd newaddon

# initialize as new git repo
git init .

# add CC addon_common as submodule
git submodule add git@github.com:CGCookie/addon_common.git addon_common
cd addon_common
git checkout b280  # <-- important for b280!
```

### Updating CookieCutter submodule

```
# update CC addon_common
cd addon_common
git pull
```




## resources

- Blender Conference 2018 workshop [slides](https://gfx.cse.taylor.edu/courses/bcon18/index.md.html?scale) and [presentation](https://www.youtube.com/watch?v=YSHdSNhMO1c)
