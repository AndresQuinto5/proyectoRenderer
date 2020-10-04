#Andres Emilio Quinto Villagran - 18288
#Proyecto renderer
#Codigo basado en lo trabajado en clase

import struct
from math import sin, cos, sinh, cosh
import random
from obj import OBJ
from collections import namedtuple
import numpy as np

class Texture(object):
    """
	"""

    def __init__(self, filename):
        """
        File init
		"""
        self.__filename = filename
        self.__active_texture = None
        self.load()

    def load(self):
        """
        Texture Loading
		"""
        print("Cargando Textura --->")
        self.__active_texture = BMP(0, 0)
        try:
            self.__active_texture.load(self.__filename)
        except:
            print("Textura cargada.")
            self.__active_texture = None

    def write(self):
        """
        Texture Write
		"""
        self.__active_texture.write(
            self.__filename[: len(self.__filename) - 4] + "texture.bmp"
        )

    def get_color(self, tx, ty, intensity=1):
        """
        Get color definition
		"""
        x = (
            self.__active_texture.width - 1
            if ty == 1
            else int(ty * self.__active_texture.width)
        )
        y = (
            self.__active_texture.height - 1
            if tx == 1
            else int(tx * self.__active_texture.height)
        )
        return bytes(
            map(
                lambda b: round(b * intensity) if b * intensity > 0 else 0,
                self.__active_texture.framebuffer[y][x],
            )
        )

    def has_valid_texture(self):
        return self.__active_texture != None

#Shaders Code Start

#Here is the code of PHONG shader, for my uses is called colores1 
def colores1(render, light=(0, 0, 1), bary=(1, 1, 1), normals=0, base_color=(1, 1, 1)):

    # barycentric
    w, v, u = bary
    # normals
    nA, nB, nC = normals
    b, g, r = base_color

    iA, iB, iC = [dot(n, light) for n in (nA, nB, nC)]
    intensity = w * iA + v * iB + u * iC

    b *= intensity
    g *= intensity
    r *= intensity

    if intensity > 0:
        return render.set_color(r, g, b)
    else:
        return render.set_color(0, 0, 0)

#Here is the code of INVERSE shader, for my uses is called colores2 
def colores2(render, light=(0, 0, 1), bary=(1, 1, 1), normals=0, base_color=(1, 1, 1)):

    # barycentric
    w, v, u = bary
    # normals
    nA, nB, nC = normals
    b, g, r = base_color

    iA, iB, iC = [dot(n, light) for n in (nA, nB, nC)]
    intensity = w * iA + v * iB + u * iC

    b *= (1 - intensity)
    g *= (1 - intensity)
    r *= (1 - intensity)

    if intensity > 0:
        return render.set_color(r, g, b)
    else:
        return render.set_color(0, 0, 0)

#Here is the code of BRIGHT shader, for my uses is called colores3 
def colores3(render, light=(0, 0, 1), bary=(1, 1, 1), normals=0, base_color=(1, 1, 1)):
    """
	Shader to make brighter
	"""
    base_color = (1, 0.85, 0)
    w, v, u = bary
    nA, nB, nC = normals
    light = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))

    iA = dot(nA, light)
    iB = dot(nB, nA)
    iC = dot(nC, nB)

    b, g, r = base_color
    intensity = w * iA + v * iB + u * iC

    if intensity <= 0:
        intensity = 0.85
    elif intensity < 0.25:
        intensity = 0.75
    elif intensity < 0.75:
        intensity = 0.55
    elif intensity < 1:
        r, g, b = 1, 1, 0
        intensity = 0.80

    b *= intensity
    g *= intensity
    r *= intensity

    return render.set_color(r, g, b)

#Here is the code of SMOOTH shader, for my uses is called colores4 
def colores4(render, light=(0, 0, 1), bary=(1, 1, 1), normals=0, base_color=(1, 1, 1)):
    """
	Shader to smooth surfaces
	"""
    # barycentric
    w, v, u = bary
    # normals
    nA, nB, nC = normals
    # light intensity
    light = (1, 0, 1)

    b, g, r = base_color
    iA, iB, iC = [dot(n, light) for n in (nA, nB, nC)]
    intensity = w * iA + v * iB + u * iC

    b *= intensity
    g *= intensity
    r *= intensity

    return render.set_color(r, g, b)

#Here is the code of GRADUAL TOON shader, for my uses is called colores5 
def colores5(render, light=(0, 0, 1), bary=(1, 1, 1), normals=0, base_color=(1, 1, 1)):

    u, v, w = bary
    nA, nB, nC = normals
    b, g, r = base_color

    iA, iB, iC = [dot(n, light) for n in (nA, nB, nC)]
    intensity = w * iA + v * iB + u * iC

    b *= intensity
    g *= intensity
    r *= intensity

    if intensity > 0:
        return render.set_color(round(r,1), round(g,1), round(b,1))
    else:
        return render.set_color(0, 0, 0)

#Box bounding
def bbox(*vertex_list):
    """
	Smallest possible bounding box
	"""
    xs = [vertices[0] for vertices in vertex_list]
    ys = [vertices[1] for vertices in vertex_list]
    xs.sort()
    ys.sort()
    return (xs[0], ys[0]), (xs[-1], ys[-1])

#Coords
def barycentric(A, B, C, x, y):
    """
	Gets barycentric coords
	"""
    v1 = (C[0] - A[0], B[0] - A[0], A[0] - x)
    v2 = (C[1] - A[1], B[1] - A[1], A[1] - y)
    bary = cross(v1, v2)
    if abs(bary[2]) < 1:
        return -1, -1, -1

    return (1 - (bary[0] + bary[1]) / bary[2], bary[1] / bary[2], bary[0] / bary[2])


def norm(v0):
    v = length(v0)
    if not v:
        return [0, 0, 0]
    return [v0[0] / v, v0[1] / v, v0[2] / v]


def length(v0):
    return (v0[0] ** 2 + v0[1] ** 2 + v0[2] ** 2) ** 0.5


def point_inside_polygon(x, y, vertex_list):
    """
	Checks if (x, y) point is inside polygon 
	"""
    even_accumulator = 0
    point_1 = vertex_list[0]
    n = len(vertex_list)
    for i in range(n + 1):
        point_2 = vertex_list[i % n]
        if y > min(point_1[1], point_2[1]):
            if y <= max(point_1[1], point_2[1]):
                if point_1[1] != point_2[1]:
                    xinters = (y - point_1[1]) * (point_2[0] - point_1[0]) / (
                        point_2[1] - point_1[1]
                    ) + point_1[0]
                    if point_1[0] == point_2[0] or x <= xinters:
                        even_accumulator += 1
        point_1 = point_2
    if even_accumulator % 2 == 0:
        return False
    else:
        return True


def dot(v0, v1):
    """
	Dot product
	"""
    return v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2]


def cross(v0, v1):
    """
	Cross product
	"""
    return [
        v0[1] * v1[2] - v0[2] * v1[1],
        v0[2] * v1[0] - v0[0] * v1[2],
        v0[0] * v1[1] - v0[1] * v1[0],
    ]


def vector(p, q):
    """
	Vector pq
	"""
    return [q[0] - p[0], q[1] - p[1], q[2] - p[2]]


def sub(v0, v1):
    """
	Vector subtraction
	"""
    return [v0[0] - v1[0], v0[1] - v1[1], v0[2] - v1[2]]


def get_zplane_value(vertex_list, x, y):
    """
	Gets z-coord in (x,y,z) found in the plane that passes through the first 3 points of vertex_list
	"""
    pq = vector(vertex_list[0], vertex_list[1])
    pr = vector(vertex_list[0], vertex_list[2])
    normal = cross(pq, pr)
    if normal[2]:
        z = (
            (normal[0] * (x - vertex_list[0][0]))
            + (normal[1] * (y - vertex_list[0][1]))
            - (normal[2] * vertex_list[0][2])
        ) / (-normal[2])
        return z
    else:
        return -float("inf")


class Matrix(object):
    def __init__(self, data):
        self.data = data
        self.row = len(data)
        self.col = len(data[0])

    def __mul__(self, m2):
        result = []
        for i in range(self.row):
            result.append([])
            for j in range(m2.col):
                result[-1].append(0)

        for i in range(self.row):
            for j in range(m2.col):
                for k in range(m2.row):
                    result[i][j] += self.data[i][k] * m2.data[k][j]

        return Matrix(result)

    def to_list(self):
        return self.data

class BMP(object):
    def __init__(self, width, height):
        self.width = abs(int(width))
        self.height = abs(int(height))
        self.framebuffer = []
        self.zbuffer = []
        self.clear()

    def clear(self, r=0, g=0, b=0):
        self.framebuffer = [
            [self.color(r, g, b) for x in range(self.width)] for y in range(self.height)
        ]

        self.zbuffer = [
            [-float("inf") for x in range(self.width)] for y in range(self.height)
        ]

    def color(self, r=0, g=0, b=0):
        if r > 255 or g > 255 or b > 255 or r < 0 or g < 0 or b < 0:
            r = 0
            g = 0
            b = 0
        return bytes([b, g, r])

    def point(self, x, y, color):
        if x < self.width and y < self.height:
            try:
                self.framebuffer[x][y] = color
            except Exception as e:
                pass

    def write(self, filename, zbuffer=False):
        print("Writing " + filename +" bmp file")
        BLACK = self.color(0, 0, 0)
        import os

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        file = open(filename, "bw")

        # File Header (14 bytes)
        file.write(self.__char("B"))
        file.write(self.__char("M"))  # BM
        file.write(self.___dword(14 + 40 + self.width * self.height * 3))  # File size
        file.write(self.___dword(0))
        file.write(self.___dword(14 + 40))

        # Image Header (14 bytes)
        file.write(self.___dword(40))
        file.write(self.___dword(self.width))
        file.write(self.___dword(self.height))
        file.write(self.___word(1))
        file.write(self.___word(24))
        file.write(self.___dword(0))
        file.write(self.___dword(self.width * self.height * 3))
        file.write(self.___dword(0))
        file.write(self.___dword(0))
        file.write(self.___dword(0))
        file.write(self.___dword(0))

        for x in range(self.width):
            for y in range(self.height):
                if x < self.width and y < self.height:
                    if zbuffer:
                        if self.zbuffer[y][x] == -float("inf"):
                            file.write(BLACK)
                        else:
                            z = abs(int(self.zbuffer[y][x] * 255))
                            file.write(self.color(z, z, z))
                    else:
                        file.write(self.framebuffer[y][x])
                else:
                    file.write(self.__char("c"))

        file.close()

    def load(self, filename):
        file = open(filename, "rb")
        file.seek(10)
        headerSize = struct.unpack("=l", file.read(4))[0]
        file.seek(18)

        self.width = struct.unpack("=l", file.read(4))[0]
        self.height = struct.unpack("=l", file.read(4))[0]
        self.clear()
        for y in range(self.height):
            for x in range(self.width):
                if x < self.width and y < self.height:
                    r, g, b = ord(file.read(1)), ord(file.read(1)), ord(file.read(1))
                    self.point(x, y, self.color(r, b, g))
        file.close()

    def __padding(self, base, c):
        if c % base == 0:
            return c
        else:
            while c % base:
                c += 1
            return c

    def __char(self, c):
        return struct.pack("=c", c.encode("ascii"))

    def ___word(self, c):
        return struct.pack("=h", c)

    def ___dword(self, c):
        return struct.pack("=l", c)

    def get_zbuffer_value(self, x, y):
        if x < self.width and y < self.height:
            return self.zbuffer[y][x]
        else:
            return -float("inf")

    def set_zbuffer_value(self, x, y, z):
        if x < self.width and y < self.height:
            self.zbuffer[y][x] = z
            return 1
        else:
            return 0

class Render(object):
    def __init__(self):
        self.__render = BMP(0, 0)
        self.__viewport_start = (0, 0)
        self.__viewport_size = (0, 0)
        self.__color = self.__render.color(255, 255, 255)
        self.__filename = "out.bmp"
        self.__obj = None
        self.__active_texture = None

    def create_window(self, width, height):
        self.__render = BMP(width, height)
        self.__viewport_size = (width, height)

    def viewport(self, x, y, width, height):
        self.__viewport_start = (x, y)
        self.__viewport_size = (width, height)

    def clear(self):
        self.__render.clear()

    def clear_color(self, r, g, b):
        self.__render.clear(int(255 * r), int(255 * g), int(255 * b))

    def vertex(self, x, y):
        viewport_x = int(
            self.__viewport_size[0] * (x + 1) * (1 / 2) + self.__viewport_start[0]
        )
        viewport_y = int(
            self.__viewport_size[1] * (y + 1) * (1 / 2) + self.__viewport_start[1]
        )
        self.__render.point(viewport_x, viewport_y, self.__color)

    def flood_vertex(self, x, y):
        viewport_x = int(
            self.__viewport_size[0] * (x + 1) * (1 / 2) + self.__viewport_start[0]
        )
        viewport_y = int(
            self.__viewport_size[1] * (y + 1) * (1 / 2) + self.__viewport_start[1]
        )
        self.__render.point(viewport_x, viewport_y, self.__color)
        self.__render.point(viewport_x, viewport_y + 1, self.__color)
        self.__render.point(viewport_x + 1, viewport_y, self.__color)
        self.__render.point(viewport_x + 1, viewport_y + 1, self.__color)

    def set_color(self, r, g, b):
        self.__color = self.__render.color(int(255 * r), int(255 * g), int(255 * b))
        return self.__render.color(int(255 * r), int(255 * g), int(255 * b))

    def set_background(self, background):
        self.__active_texture = Texture(background)

        for x in range(0, self.__viewport_size[0]):
            for y in range(0, self.__viewport_size[1]):
                custom = self.__active_texture.get_color((x/self.__viewport_size[0]), (y/self.__viewport_size[1]))
                self.__render.point(x, y, custom)


    def finish(self):
        self.__render.write(self.__filename)

    def line(self, xo, yo, xf, yf):
        """
		Draws a line between two coords
		"""
        x1 = int(self.__viewport_size[0] * (xo + 1) * (1 / 2) + self.__viewport_start[0])
        y1 = int(self.__viewport_size[1] * (yo + 1) * (1 / 2) + self.__viewport_start[1])
        x2 = int(self.__viewport_size[0] * (xf + 1) * (1 / 2) + self.__viewport_start[0])
        y2 = int(self.__viewport_size[1] * (yf + 1) * (1 / 2) + self.__viewport_start[1])
        dy = abs(y2 - y1)
        dx = abs(x2 - x1)
        steep = dy > dx
        if steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        dy = abs(y2 - y1)
        dx = abs(x2 - x1)
        offset = 0
        threshold = dx
        y = y1
        for x in range(x1, x2 + 1):
            if steep:
                self.__render.point(y, x, self.__color)
            else:
                self.__render.point(x, y, self.__color)

            offset += dy * 2
            if offset >= threshold:
                y += 1 if y1 < y2 else -1
                threshold += 2 * dx

    def set_filename(self, filename):
        self.__filename = filename

    def load_obj(
        self,
        filename,
        translate=(0, 0, 0),
        scale=(1, 1, 1),
        fill=True,
        textured=None,
        rotate=(0, 0, 0),
        shader=None,
    ):
        """
		Loads OBJ file
		"""
        self.model_matrix(translate, scale, rotate)
        self.__obj = OBJ(filename)
        self.__obj.load()
        light = norm((0, 0, 1))
        faces = self.__obj.get_faces()
        vertices = self.__obj.get_vertices()
        materials = self.__obj.get_materials()
        text_vertices = self.__obj.get_texture_vertices()
        normals = self.__obj.get_normals()
        material_faces = self.__obj.get_material_faces()
        self.__active_texture = Texture(textured)

        print("Rendering " + filename + " ...")

        if materials:
            for mats in material_faces:
                start, stop = mats[0]
                color = materials[mats[1]].diffuse_color
                for index in range(start, stop):
                    face = faces[index]
                    vcount = len(face)

                    if vcount == 3:
                        f1 = face[0][0] - 1
                        f2 = face[1][0] - 1
                        f3 = face[2][0] - 1

                        a = self.transform(vertices[f1])
                        b = self.transform(vertices[f2])
                        c = self.transform(vertices[f3])

                        if shader:
                            t1 = face[0][1] - 1
                            t2 = face[1][1] - 1
                            t3 = face[2][1] - 1
                            nA = normals[t1]
                            nB = normals[t2]
                            nC = normals[t3]
                            self.triangle(
                                a,
                                b,
                                c,
                                base_color=color,
                                shader=shader,
                                normals=(nA, nB, nC),
                            )
                        else:
                            normal = norm(cross(sub(b, a), sub(c, a)))
                            intensity = dot(normal, light)

                            if not self.__active_texture.has_valid_texture():
                                if intensity < 0:
                                    continue
                                self.triangle(
                                    a,
                                    b,
                                    c,
                                    color=self.set_color(
                                        color[0] * intensity,
                                        color[1] * intensity,
                                        color[2] * intensity,
                                    ),
                                )

        else:
            print("No materials")
            for face in faces:
                vcount = len(face)

                if vcount == 3:
                    f1 = face[0][0] - 1
                    f2 = face[1][0] - 1
                    f3 = face[2][0] - 1

                    a = self.transform(vertices[f1])
                    b = self.transform(vertices[f2])
                    c = self.transform(vertices[f3])

                    if shader:
                        nA = normals[f1]
                        nB = normals[f2]
                        nC = normals[f3]
                        self.triangle(
                            a,
                            b,
                            c,
                            base_color=color,
                            shader=shader,
                            normals=(nA, nB, nC),
                        )
                    else:

                        normal = norm(cross(sub(b, a), sub(c, a)))
                        intensity = dot(normal, light)

                        if not self.__active_texture.has_valid_texture():
                            if intensity < 0:
                                continue
                            self.triangle(
                                a,
                                b,
                                c,
                                color=self.set_color(intensity, intensity, intensity),
                            )
                        else:
                            if self.__active_texture.has_valid_texture():
                                t1 = face[0][-1] - 1
                                t2 = face[1][-1] - 1
                                t3 = face[2][-1] - 1
                                tA = text_vertices[t1]
                                tB = text_vertices[t2]
                                tC = text_vertices[t3]
                                self.triangle(
                                    a,
                                    b,
                                    c,
                                    texture=self.__active_texture.has_valid_texture(),
                                    texture_coords=(tA, tB, tC),
                                    intensity=intensity,
                                )
                else:
                    f1 = face[0][0] - 1
                    f2 = face[1][0] - 1
                    f3 = face[2][0] - 1
                    f4 = face[3][0] - 1

                    vertex_list = [
                        self.transform(vertices[f1]),
                        self.transform(vertices[f2]),
                        self.transform(vertices[f3]),
                        self.transform(vertices[f4]),
                    ]
                    normal = norm(
                        cross(
                            sub(vertex_list[0], vertex_list[1]),
                            sub(vertex_list[1], vertex_list[2]),
                        )
                    )
                    intensity = dot(normal, light)
                    A, B, C, D = vertex_list
                    if not textured:
                        if intensity < 0:
                            continue
                        self.triangle(
                            A,
                            B,
                            C,
                            color=self.set_color(intensity, intensity, intensity),
                        )
                        self.triangle(
                            A,
                            C,
                            D,
                            color=self.set_color(intensity, intensity, intensity),
                        )
                    else:
                        if self.__active_texture.has_valid_texture():
                            t1 = face[0][-1] - 1
                            t2 = face[1][-1] - 1
                            t3 = face[2][-1] - 1
                            t4 = face[3][-1] - 1
                            tA = text_vertices[t1]
                            tB = text_vertices[t2]
                            tC = text_vertices[t3]
                            tD = text_vertices[t4]
                            self.triangle(
                                A,
                                B,
                                C,
                                texture=self.__active_texture.has_valid_texture(),
                                texture_coords=(tA, tB, tC),
                                intensity=intensity,
                            )
                            self.triangle(
                                A,
                                C,
                                D,
                                texture=self.__active_texture.has_valid_texture(),
                                texture_coords=(tA, tC, tD),
                                intensity=intensity,
                            )

    def triangle(
        self,
        A,
        B,
        C,
        color=None,
        texture=None,
        texture_coords=(),
        intensity=1,
        normals=None,
        shader=None,
        base_color=(1, 1, 1),
    ):
        """
		Draws a triangle ABC
		"""
        bbox_min, bbox_max = bbox(A, B, C)

        for x in range(bbox_min[0], bbox_max[0] + 1):
            for y in range(bbox_min[1], bbox_max[1] + 1):
                w, v, u = barycentric(A, B, C, x, y)
                if w < 0 or v < 0 or u < 0:
                    continue
                if texture:
                    tA, tB, tC = texture_coords
                    tx = tA[0] * w + tB[0] * v + tC[0] * u
                    ty = tA[1] * w + tB[1] * v + tC[1] * u
                    color = self.__active_texture.get_color(tx, ty, intensity)
                elif shader:
                    color = shader(
                        self, bary=(w, u, v), normals=normals, base_color=base_color
                    )
                z = A[2] * w + B[2] * v + C[2] * u
                if x < 0 or y < 0:
                    continue
                if z > self.__render.get_zbuffer_value(x, y):
                    self.__render.point(x, y, color)
                    self.__render.set_zbuffer_value(x, y, z)

    def load(
        self,
        filename,
        translate=(0, 0, 0),
        scale=(1, 1, 1),
        fill=True,
        textured=None,
        rotate=(0, 0, 0),
    ):
        """
		Carga archivos Obj como wireframe
		"""
        light = (0, 0, 1)
        self.model_matrix(translate, scale, rotate)
        self.__obj = OBJ(filename)
        self.__obj.load()
        vertices = self.__obj.get_vertices()
        faces = self.__obj.get_faces()
        normals = self.__obj.get_normals()
        materials = self.__obj.get_materials()
        text_vertices = self.__obj.get_texture_vertices()
        print("Rendering " + filename + " (as wireframe)...")

        if materials and not textured:
            material_index = self.__obj.get_material_faces()
            for mat in material_index:
                diffuse_color = materials[mat[1]].diffuse_color
                for i in range(mat[0][0], mat[0][1]):
                    coord_list = []
                    texture_coords = []
                    for face in faces[i]:
                        coo = (
                            (vertices[face[0] - 1][0] + translate[0]) * scale[0],
                            (vertices[face[0] - 1][1] + translate[1]) * scale[1],
                            (vertices[face[0] - 1][2] + translate[2]) * scale[2],
                        )
                        coord_list.append(coo)
                    if fill:
                        curr_intensity = dot(normals[face[1] - 1], light)
                        if curr_intensity < 0:
                            continue
                        self.fill_polygon(
                            coord_list,
                            color=(
                                curr_intensity * diffuse_color[0],
                                curr_intensity * diffuse_color[1],
                                curr_intensity * diffuse_color[2],
                            ),
                        )
                    else:
                        self.draw_polygon(coord_list)

        elif textured and not materials:
            for face in faces:
                coord_list = []
                texture_coords = []
                for vertexN in face:
                    coo = (
                        (vertices[vertexN[0] - 1][0] + translate[0]) * scale[0],
                        (vertices[vertexN[0] - 1][1] + translate[1]) * scale[1],
                        (vertices[vertexN[0] - 1][2] + translate[2]) * scale[2],
                    )
                    coord_list.append(coo)
                    if len(vertexN) > 2:
                        text = (
                            (text_vertices[vertexN[2] - 1][0] + translate[0]) * scale[0],
                            (text_vertices[vertexN[2] - 1][1] + translate[1]) * scale[1],
                        )
                        texture_coords.append(text)
                if fill:
                    curr_intensity = dot(normals[vertexN[1] - 1], light)
                    if curr_intensity < 0:
                        continue
                    self.fill_polygon(
                        coord_list,
                        intensity=curr_intensity,
                        texture=textured,
                        texture_coords=texture_coords,
                    )
                else:
                    self.draw_polygon(coord_list)
                coord_list = []
        else:
            for face in faces:
                coord_list = []
                texture_coords = []
                for vertexN in face:
                    coo = vertices[vertexN[0] - 1]
                    coord_list.append(coo)
                if fill:
                    curr_intensity = dot(normals[vertexN[1] - 1], light)
                    if curr_intensity < 0:
                        continue
                    self.fill_polygon(coord_list, color=(curr_intensity, curr_intensity, curr_intensity))
                else:
                    self.draw_polygon(coord_list)
                coord_list = []

    def draw_polygon(self, vertex_list):
        """
		Draws a polygon
		"""
        for i in range(len(vertex_list)):
            if i == len(vertex_list) - 1:
                start = vertex_list[i]
                final = vertex_list[0]
            else:
                start = vertex_list[i]
                final = vertex_list[i + 1]
            self.line(start[0], start[1], final[0], final[1])

    def fill_polygon(
        self,
        vertex_list,
        color=None,
        texture=None,
        intensity=1,
        texture_coords=(),
        zVal=True,
    ):
        """
		Draws and fills a polygon
		"""
        curr_intensity = intensity
        if not texture:
            color = (
                self.__color
                if color == None
                else self.__render.color(
                    int(255 * color[0]), int(255 * color[1]), int(255 * color[2])
                )
            )
        else:
            if self.__active_texture == None:
                text = Texture(texture)
                self.__active_texture = text
            else:
                text = self.__active_texture
        start_x = sorted(vertex_list, key=lambda tup: tup[0])[0][0]
        final_x = sorted(vertex_list, key=lambda tup: tup[0], reverse=True)[0][0]

        start_y = sorted(vertex_list, key=lambda tup: tup[1])[0][1]
        final_y = sorted(vertex_list, key=lambda tup: tup[1], reverse=True)[0][1]

        start_x = int(
            self.__viewport_size[0] * (start_x + 1) * (1 / 2) + self.__viewport_start[0]
        )
        final_x = int(
            self.__viewport_size[0] * (final_x + 1) * (1 / 2) + self.__viewport_start[0]
        )

        start_y = int(
            self.__viewport_size[0] * (start_y + 1) * (1 / 2) + self.__viewport_start[0]
        )
        final_y = int(
            self.__viewport_size[0] * (final_y + 1) * (1 / 2) + self.__viewport_start[0]
        )
        for x in range(start_x, final_x + 1):
            for y in range(start_y, final_y + 1):
                is_inside = point_inside_polygon(self.normalize_x(x), self.normalize_y(y), vertex_list)
                if is_inside:
                    if texture:
                        A = (
                            self.normalize_inv_x(vertex_list[0][0]),
                            self.normalize_inv_x(vertex_list[0][1]),
                        )
                        B = (
                            self.normalize_inv_x(vertex_list[1][0]),
                            self.normalize_inv_x(vertex_list[1][1]),
                        )
                        C = (
                            self.normalize_inv_x(vertex_list[2][0]),
                            self.normalize_inv_x(vertex_list[2][1]),
                        )
                        w, v, u = barycentric(A, B, C, x, y)
                        A = texture_coords[0]
                        B = texture_coords[1]
                        C = texture_coords[2]
                        tx = A[0] * w + B[0] * v + C[0] * u
                        ty = A[1] * w + B[1] * v + C[1] * u
                        color = text.get_color(tx, ty, intensity=curr_intensity)
                    z = get_zplane_value(vertex_list, x, y)
                    if z > self.__render.get_zbuffer_value(x, y):
                        self.__render.point(x, y, color)
                        self.__render.set_zbuffer_value(x, y, z)

    def normalize_x(self, x):
        """
		Normalizes x coord
		"""
        normalize_x = ((2 * x) / self.__viewport_size[0]) - self.__viewport_start[0] - 1
        return normalize_x

    def normalize_y(self, y):
        """
		Normalizes y coord
		"""
        normalize_y = ((2 * y) / self.__viewport_size[1]) - self.__viewport_start[1] - 1
        return normalize_y

    def normalize_inv_x(self, x):
        """
		Normalizes x inv coord
		"""
        normalize_x = int(self.__viewport_size[0] * (x + 1) * (1 / 2) + self.__viewport_start[0])
        return normalize_x

    def normalize_inv_y(self, y):
        """
		Normalizes y inv coord
		"""
        normalize_y = int(self.__viewport_size[0] * (y + 1) * (1 / 2) + self.__viewport_start[0])
        return normalize_y

    def render_zbuffer(self, filename=None):
        """
		Renders the z buffer stored
		"""
        if filename == None:
            filename = self.__filename.split(".")[0] + "ZBuffer.bmp"
        self.__render.write(filename, zbuffer=True)

    def model_matrix(self, translate=(0, 0, 0), scale=(1, 1, 1), rotate=(0, 0, 0)):
        """
		Generates model matrix
		"""
        translation_matrix = Matrix(
            [
                [1, 0, 0, translate[0]],
                [0, 1, 0, translate[1]],
                [0, 0, 1, translate[2]],
                [0, 0, 0, 1],
            ]
        )
        a = rotate[0]
        rotation_matrix_x = Matrix(
            [
                [1, 0, 0, 0],
                [0, cos(a), -sin(a), 0],
                [0, sin(a), cos(a), 0],
                [0, 0, 0, 1],
            ]
        )
        a = rotate[1]
        rotation_matrix_y = Matrix(
            [
                [cos(a), 0, sin(a), 0],
                [0, 1, 0, 0],
                [-sin(a), 0, cos(a), 0],
                [0, 0, 0, 1],
            ]
        )
        a = rotate[2]
        rotation_matrix_z = Matrix(
            [
                [cos(a), -sin(a), 0, 0],
                [sin(a), cos(a), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rotation_matrix = rotation_matrix_x * rotation_matrix_y * rotation_matrix_z
        scale_matrix = Matrix(
            [
                [scale[0], 0, 0, 0],
                [0, scale[1], 0, 0],
                [0, 0, scale[2], 0],
                [0, 0, 0, 1],
            ]
        )
        self.Model = translation_matrix * rotation_matrix * scale_matrix

    def view_matrix(self, x, y, z, center):
        """
		Generates view matrix
		"""
        m = Matrix(
            [
                [x[0], x[1], x[2], 0],
                [y[0], y[1], y[2], 0],
                [z[0], z[1], z[2], 0],
                [0, 0, 0, 1],
            ]
        )
        o = Matrix(
            [
                [1, 0, 0, -center[0]],
                [0, 1, 0, -center[1]],
                [0, 0, 1, -center[2]],
                [0, 0, 0, 1],
            ]
        )
        self.View = m * o

    def projection_matrix(self, coeff):
        """
		Generates projection matrix
		"""
        self.Projection = Matrix(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, coeff, 1]]
        )

    def viewport_matrix(self, x=0, y=0):
        """
		Generates viewport matrix
		"""
        self.Viewport = Matrix(
            [
                [self.__render.width / 2, 0, 0, x + self.__render.width / 2],
                [0, self.__render.height / 2, 0, y + self.__render.height / 2],
                [0, 0, 128, 128],
                [0, 0, 0, 1],
            ]
        )
# Renderizado del sátelite que orbita por la luna
#render.load_obj(
 #   "./Modelos/space_station.obj",
  #  translate=(0.25, -0.75, 0.3),
   # scale=(0.03, 0.03, 0.03),
    #rotate=(-0.50, -0.80, 0.2),
    #fill=True,
    #shader=colores2,
#)
    def look_at(self, eye, center, up):
        """
		Defines camera position
		"""
        z = norm(sub(eye, center))
        x = norm(cross(up, z))
        y = norm(cross(z, x))
        self.view_matrix(x, y, z, center)
        self.projection_matrix(-1 / length(sub(eye, center)))
        self.viewport_matrix()

    def transform(self, vertices):
        """
		Transforms with augmented matrix
		"""
        augmented = Matrix([[vertices[0]], [vertices[1]], [vertices[2]], [1]])
        transformed_vertex = (
            self.Viewport * self.Projection * self.View * self.Model * augmented
        )
        transformed_vertex = transformed_vertex.to_list()
        transformed = [
            round(transformed_vertex[0][0] / transformed_vertex[3][0]),
            round(transformed_vertex[1][0] / transformed_vertex[3][0]),
            round(transformed_vertex[2][0] / transformed_vertex[3][0]),
        ]
        return transformed