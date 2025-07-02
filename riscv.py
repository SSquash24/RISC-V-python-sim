# riscv.py: Implements the risc-v sim as various classes, per type of RISCV
# classes:
#   - RV32I

import yamls.read_yaml as yamls # to decode instrs

class RISCBase:

    def set_regs(self, reg, val):
        assert(0 <= reg < self.reg_count)
        if reg != 0:  # reg 0 never changes
            self.regs[reg] = val

    def __init__(self, word_size, mem_size, code, reg_count=32, debug=False):
        """
        Base class for RISCV simulators to build off, should not be instantiated directly.
        :param word_size: Size of a single instruction word in bytes.
        :param mem_size: Amount of memory in words.
        :param code: *Compiled* code in a list of words format, will be placed starting at mem addr 0.
        :param reg_count: Number of registers, default 32. Note reg 0 is hardwired to value 0.
        :param debug: Should debug statements be printed?
        """
        assert(type(debug) is bool)
        self.debug = debug
        self.mem_size = mem_size
        self.code = code
        self.word_size = word_size

        self.mem = self.code + [0] * (self.mem_size - len(self.code))

        self.reg_count = reg_count
        self.regs = [0 for _ in range(reg_count)]

        self.pc = 0 # pc register store address of next instr (address is in bytes)

    def step(self):
        if self.debug:
            print("RISCV command: ")
        instr = self.mem[self.pc]
        opcode = instr % 128 # last 7 bits
        return instr, opcode


class RV32I(RISCBase):
    def __init__(self, mem_size, code, debug=False):
        super().__init__(4, mem_size, code, debug=debug)

    def step(self):
        instr, opcode = super().step()
        decoded_opcode = yamls.opcodes[opcode]
        # TODO actually decode & run instr
        self.pc += 4