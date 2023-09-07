from collections import namedtuple

# Define a named tuple called 'Label' with fields 'name' and 'color' for defining classes.
Label = namedtuple('Label', ['name', 'color'])

# Define label definitions using the named tuple 'Label' to define classes
label_defs = [
    Label('road',           (  64,  32,  32 ) ),  # Road (all parts, anywhere nobody would look at you funny for driving)
    Label('lanes',          ( 255,   0,   0 ) ),  # Lane markings (don't include non-lane markings like turn arrows and crosswalks)
    Label('undrivable',     ( 128, 128,  96 ) ),  # Undrivable areas
    Label('movable',        (   0, 255, 102 ) ),  # Movable objects (vehicles and people/animals)
    Label('my_car',         ( 204,   0, 255 ) )]  # My car (and anything inside it, including wires, mounts, etc. No reflections)

def num_classes():
    # Return the number of label definitions or classes.
    return len(label_defs)

