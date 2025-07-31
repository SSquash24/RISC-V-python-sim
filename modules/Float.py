from RISC_modules import *
from yamls.read_yaml import read_yaml

class Float(Module):

    opcodes, inv_opcodes = read_yaml('float.yaml')