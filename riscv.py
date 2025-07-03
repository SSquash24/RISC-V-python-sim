# riscv.py: Implements the risc-v sim as various classes, per type of RISCV
# classes:
#   - RV32I

import compiler
from compiler import Comp_RV32I


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
        assert all(len(i) == 8 * word_size for i in code)
        self.debug = debug
        self.mem_size = mem_size
        self.code = code
        self.word_size = word_size

        self.mem = self.code + ['0' * 8 * word_size] * (self.mem_size - len(self.code))

        self.reg_count = reg_count
        self.regs = [0 for _ in range(reg_count)]

        self.pc = 0 # pc register store address of next instr (address is in bytes)

    def step(self):
        if self.debug:
            print("RISCV command: ")
        return self.mem[self.pc // 4]


class RV32I(RISCBase):
    def __init__(self, mem_size, code, debug=False):
        super().__init__(4, mem_size, code, debug=debug)

    def step(self):

        def unsigned(num):
            return num % (1 << 32)
        def revert(num, size=32):
            return num if num < (1 << size-1) else (1 << size) - num

        instr = super().step()
        instr, *args = Comp_RV32I.decompile_instr(instr)
        print(instr, args)
        match instr:
            # OPS
            case 'add':
                self.regs[args[0]] = self.regs[args[1]] + self.regs[args[2]]
            case 'sub':
                self.regs[args[0]] = self.regs[args[1]] - self.regs[args[2]]
            case 'xor':
                self.regs[args[0]] = self.regs[args[1]] ^ self.regs[args[2]]
            case 'or':
                self.regs[args[0]] = self.regs[args[1]] | self.regs[args[2]]
            case 'and':
                self.regs[args[0]] = self.regs[args[1]] & self.regs[args[2]]
            case 'sll':
                self.regs[args[0]] = self.regs[args[1]] << self.regs[args[2]]
            case 'srl':
                self.regs[args[0]] = revert(unsigned(self.regs[args[1]]) >> self.regs[args[2]])
            case 'sra':
                self.regs[args[0]] = self.regs[args[1]] >> self.regs[args[2]]
            case 'slt':
                self.regs[args[0]] = 1 if self.regs[args[1]] < self.regs[args[2]] else 0
            case 'sltu':
                self.regs[args[0]] = 1 if unsigned(self.regs[args[1]]) < unsigned(self.regs[args[2]]) else 0
            # IMM-OPS
            case 'addi':
                self.regs[args[0]] = self.regs[args[1]] + args[2]
            case 'xori':
                self.regs[args[0]] = self.regs[args[1]] ^ args[2]
            case 'ori':
                self.regs[args[0]] = self.regs[args[1]] | args[2]
            case 'andi':
                self.regs[args[0]] = self.regs[args[1]] & args[2]
            case 'slli':
                self.regs[args[0]] = self.regs[args[1]] << args[2]
            case 'srli':
                self.regs[args[0]] = revert(unsigned(self.regs[args[1]]) >> args[2])
            case 'srai':
                self.regs[args[0]] = self.regs[args[1]] >> args[2]
            case 'slti':
                self.regs[args[0]] = 1 if self.regs[args[1]] < args[2] else 0
            case 'sltiu':
                self.regs[args[0]] = 1 if unsigned(self.regs[args[1]]) < unsigned(args[2]) else 0
            # LOAD
            case 'lb':
                loc, offset = divmod(self.regs[args[1] + args[2]], 4)
                self.regs[args[0]] = revert((Comp_RV32I.twos_to_int(self.mem[loc]) >> 8 * offset) % (1 << 8), size=8)
            case 'lh':
                loc, offset = divmod(self.regs[args[1] + args[2]], 4)
                assert loc % 2 == 0
                self.regs[args[0]] = revert((Comp_RV32I.twos_to_int(self.mem[loc]) >> 8 * offset) % (1<<16), size=16)
            case 'lw':
                loc, offset = divmod(self.regs[args[1] + args[2]], 4)
                assert offset == 0
                self.regs[args[0]] = Comp_RV32I.twos_to_int(self.mem[loc])
            case 'lbu':
                loc, offset = divmod(self.regs[args[1] + args[2]], 4)
                self.regs[args[0]] = (int(self.mem[loc], 2) >> 8 * offset) % (1 << 8)
            case 'lhu':
                loc, offset = divmod(self.regs[args[1] + args[2]], 4)
                assert loc % 2 == 0
                self.regs[args[0]] = revert((int(self.mem[loc], 2) >> 8 * offset) % (1 << 16))
            # STORE
            case 'sb':
                loc, offset = divmod(self.regs[args[1] + args[2]], 4)
                data = bin(unsigned(self.regs[args[0]]) % (1<<8))[2:]
                self.mem[loc] = self.mem[loc][:(3-offset)*8] + data + self.mem[loc][:(offset+1)*8]
            case 'sh':
                loc, offset = divmod(self.regs[args[1] + args[2]], 4)
                assert offset % 2 == 0
                data = bin(unsigned(self.regs[args[0]]) % (1 << 16))[2:]
                self.mem[loc] = self.mem[loc][:(1 - offset) * 16] + data + self.mem[loc][:(offset + 1) * 16]
            case 'sw':
                loc, offset = divmod(self.regs[args[1] + args[2]], 4)
                assert offset == 0
                data = Comp_RV32I.twos_comp_bin(self.regs[args[0]], 32)
                self.mem[loc] = data
            # BRANCH
            case 'beq':
                if self.regs[args[0]] == self.regs[args[1]]:
                    self.pc += args[2]-4 # -4 to counteract self.pc += 4 at end
            case 'bne':
                if self.regs[args[0]] != self.regs[args[1]]:
                    self.pc += args[2] - 4
            case 'blt':
                if self.regs[args[0]] < self.regs[args[1]]:
                    self.pc += args[2] - 4
            case 'bge':
                if self.regs[args[0]] >= self.regs[args[1]]:
                    self.pc += args[2] - 4
            case 'bltu':
                if unsigned(self.regs[args[0]]) < unsigned(self.regs[args[1]]):
                    self.pc += args[2] - 4
            case 'bgeu':
                if unsigned(self.regs[args[0]]) >= unsigned(self.regs[args[1]]):
                    self.pc += args[2] - 4
            # JUMP
            case 'jal':
                self.regs[args[0]] = self.pc+4
                self.pc += args[1]
            case 'jalr':
                self.regs[args[0]] = self.pc+4
                self.pc += self.regs[args[1]] + args[2]
            # LOAD UPPER
            case 'lui':
                self.regs[args[0]] = args[1]
            case 'auipc':
                self.regs[args[0]] = self.pc + args[1]
            # TODO ECALL, EBREAK

        self.pc += 4