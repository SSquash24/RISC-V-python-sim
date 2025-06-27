# riscv.py: Implements the risc-v sim as class RISCV
# import: from riscv import RISCV

RV_types = ["RV32I", "RV64I"]


class RISCV:


    def __init__(self, RV_type, code, debug=False):
        assert(RV_type in RV_types)
        assert(type(debug) is bool)
        self.debug = debug
        self.code = code
        self.RV_type = RV_type

    def step(self):
        if self.debug:
            print("RISCV command: ")
