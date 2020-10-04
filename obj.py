#Andres Emilio Quinto Villagran - 18288
#Proyecto renderer
#Codigo basado en lo trabajado en clase

import struct

class MTL(object):
    def __init__(self, filename):
        self.__filename = filename
        self.__file = None
        self.materials = {}
        self.read_mtl()

    def read_mtl(self):
        try:
            self.__file = open(self.__filename, "r")
            self.__mtl_file = True
        except:
            self.__mtl_file = False

    def is_file_opened(self):
        return self.__mtl_file

    