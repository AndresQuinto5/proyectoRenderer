#Andres Emilio Quinto Villagran - 18288
#Proyecto renderer
#Codigo basado en lo trabajado en clase

from numpy import random
import random
from gl import Render
from gl import colores1, colores2, colores3, colores4, colores5

render = Render()
render.create_window(1600, 1600)
render.look_at((-1, 3, 5), (0, 0, 0), (0, 1, 0))
render.create_window(1600, 1600)
render.set_filename("./output.bmp")

render.set_background('./Models/background.bmp')

# Renderización de Asteroides ubicados aleatoriamente por la parte central de la imagen.
for i in range(random.randint(2, 4)):
    x = random.uniform(-0.7, 0.7)
    y = random.uniform(-0.7, 0.7)
    render.load_obj(
        "./Models/asteroid.obj",
        translate=(round(x, 1), round(y, 1), -0.30),
        scale=(0.1, 0.1, 0.1),
        rotate=(0.35, 0.28, -0.1),
        fill=True,
        shader=colores5,
    )


# Renderizado del planeta de la parte inferior izquierda
render.load_obj(
    "./Models/planet.obj",
    translate=(-0.65, -1, -0.2),
    scale=(0.70, 0.70, 0.70),
    rotate=(0, 0, 0.5),
    fill=True,
    shader=colores5,
) 

# Renderizado del austronauta de la luna
render.load_obj(
    "./Models/astronaut.obj",
    translate=(-0.65, -1, -0.45),
    scale=(0.03, 0.03, 0.03),
    rotate=(-1, -0.75, 0.15),
    fill=True,
    shader=colores2,
)

# Renderizado del Segundo Austronauta de la luna 
render.load_obj(
    "./Models/astronaut.obj",
    translate=(-0.10, -0.45, -0.01),
    scale=(0.01, 0.01, 0.01),
    rotate=(-1, -0.75, 0.15),
    fill=True,
    shader=colores5,
)
# Renderizado del sátelite que orbita por la luna
render.load_obj(
    "./Models/space_station.obj",
    translate=(0.25, -0.75, 0.3),
    scale=(0.03, 0.03, 0.03),
    rotate=(-0.50, -0.80, 0.2),
    fill=True,
    shader=colores2,
)

# Renderizado del sátelite que orbita solo
render.load_obj(
    "./Models/space_station.obj",
    translate=(0.55, 0.75, 0),
    scale=(0.01, 0.01, 0.01),
    rotate=(-0.50, -0.80, 0.2),
    fill=True,
    shader=colores5,
)
# Renderizado de la Luna ubicado en la parte superior derecha
render.load_obj(
    "./Models/planet.obj",
    translate=(-0.75, 1, 0),
    scale=(0.20, 0.20, 0.20),
    shader=colores4,
)

render.finish()
