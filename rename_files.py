from os import *
import string
import random

Img_path = "./images/humans/"

def random_char(y):
    return ''.join(random.choice(string.ascii_letters) for _ in range(y))

def rename_files():
    for path, subdirs, files in walk(Img_path):
        for name in files:
            code = random_char(5)
            rename( (f'{path}\{name}'), (f'{path}\{code}{name}') )

rename_files()