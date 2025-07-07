# riscv.py: Implements the risc-v sim as various classes, per type of RISCV
# classes:
#   - RV32I

from compiler import Comp_RV32I, RVError


class RISCBase:

    def _set_reg(self, reg, val):
        """
        Protected method
        Use instead of manually writing to the self.reg list to neatly catch errors and deal with special registers (e.g. 0)
        :param reg: Integer val of register
        :param val: Integer value to write
        :return: None if successful, otherwise RVError exception
        """
        if not (0 <= reg < self.reg_count):
            raise RVError(f"Invalid register write: x{reg}")
        if reg != 0:  # reg 0 never changes
            self.regs[reg] = val

    def _read_reg(self, reg, signed=True):
        """
        Protected method
        Use instead of manually reading the self.reg list to neatly catch errors
        :param reg: Integer val of register
        :param signed: Read as signed value (Default) as opposed to unsigned
        :return: reg value as Integer if successful, otherwise RVError exception
        """
        if not (0 <= reg < self.reg_count):
            raise RVError(f"Invalid register read: x{reg}")
        if signed:
            return self.regs[reg]
        else:
            return self.regs[reg] % (1 << ((self.word_size*8)-1))

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
        self.status = 'RUNNING'


    def step(self):
        """
        Perform one step (instruction) of the RISCV simulation.
        :return: Name of the instruction.
        """
        if self.status != 'RUNNING':
            return None
        instr = self.mem[self.pc // 4]
        if self.debug:
            print(f"RISCV command: {int(instr, 2):#x}")
        return instr


class RV32I(RISCBase):
    # 32-bit base implementation

    def __init__(self, mem_size, code, debug=False):
        super().__init__(4, mem_size, code, debug=debug)

    def step(self): # override

        #helper func to convert from unsigned to signed
        def revert(num, size=32):
            return num if num < (1 << size-1) else (1 << size) - num

        instr = super().step()
        if instr is None:
            return None
        instr, *args = Comp_RV32I.decompile_instr(instr)
        match instr:
            # OPS
            case 'add':
                self._set_reg(args[0], self._read_reg(args[1]) + self._read_reg(args[2]))
            case 'sub':
                self._set_reg(args[0], self._read_reg(args[1]) - self._read_reg(args[2]))
            case 'xor':
                self._set_reg(args[0], self._read_reg(args[1]) ^ self._read_reg(args[2]))
            case 'or':
                self._set_reg(args[0], self._read_reg(args[1]) | self._read_reg(args[2]))
            case 'and':
                self._set_reg(args[0], self._read_reg(args[1]) & self._read_reg(args[2]))
            case 'sll':
                self._set_reg(args[0], self._read_reg(args[1]) << self._read_reg(args[2]))
            case 'srl':
                self._set_reg(args[0], revert(self._read_reg(args[1], signed=False) >> self._read_reg(args[2])))
            case 'sra':
                self._set_reg(args[0], self.regs[args[1]] >> self._read_reg(args[2]))
            case 'slt':
                self._set_reg(args[0], 1 if self.regs[args[1]] < self._read_reg(args[2]) else 0)
            case 'sltu':
                self._set_reg(args[0], 1 if self._read_reg(args[1], signed=False) < self._read_reg(args[2], signed=False) else 0)
            # IMM-OPS
            case 'addi':
                self._set_reg(args[0], self._read_reg(args[1]) + args[2])
            case 'xori':
                self._set_reg(args[0], self._read_reg(args[1]) ^ args[2])
            case 'ori':
                self._set_reg(args[0], self._read_reg(args[1]) | args[2])
            case 'andi':
                self._set_reg(args[0], self._read_reg(args[1]) & args[2])
            case 'slli':
                self._set_reg(args[0], self._read_reg(args[1]) << args[2])
            case 'srli':
                self._set_reg(args[0], revert(self._read_reg(args[1], signed=False) >> args[2]))
            case 'srai':
                self._set_reg(args[0], self._read_reg(args[1]) >> args[2])
            case 'slti':
                self._set_reg(args[0], 1 if self._read_reg(args[1]) < args[2] else 0)
            case 'sltiu':
                self._set_reg(args[0], 1 if self._read_reg(args[1], signed=False) < (args[2] % (1 << 32)) else 0)
            # LOAD
            case 'lb':
                loc, offset = divmod(self._read_reg(args[1]) + args[2], 4)
                self._set_reg(args[0], revert((Comp_RV32I.twos_to_int(self.mem[loc]) >> 8 * offset) % (1 << 8), size=8))
            case 'lh':
                loc, offset = divmod(self._read_reg(args[1]) + args[2], 4)
                assert loc % 2 == 0
                self._set_reg(args[0], revert((Comp_RV32I.twos_to_int(self.mem[loc]) >> 8 * offset) % (1 << 16), size=16))
            case 'lw':
                loc, offset = divmod(self._read_reg(args[1]) + args[2], 4)
                assert offset == 0
                self._set_reg(args[0], Comp_RV32I.twos_to_int(self.mem[loc]))
            case 'lbu':
                loc, offset = divmod(self._read_reg(args[1]) + args[2], 4)
                self._set_reg(args[0], (int(self.mem[loc], 2) >> 8 * offset) % (1 << 8))
            case 'lhu':
                loc, offset = divmod(self._read_reg(args[1]) + args[2], 4)
                assert loc % 2 == 0
                self._set_reg(args[0], revert((int(self.mem[loc], 2) >> 8 * offset) % (1 << 16)))
            # STORE
            case 'sb':
                loc, offset = divmod(self._read_reg(args[1]) + args[2], 4)
                data = bin(self._read_reg(args[0], signed=False) % (1<<8))[2:]
                self.mem[loc] = self.mem[loc][:(3-offset)*8] + data + self.mem[loc][:(offset+1)*8]
            case 'sh':
                loc, offset = divmod(self._read_reg(args[1]) + args[2], 4)
                assert offset % 2 == 0
                data = bin(self._read_reg(args[0], signed=False) % (1 << 16))[2:]
                self.mem[loc] = self.mem[loc][:(1 - offset) * 16] + data + self.mem[loc][:(offset + 1) * 16]
            case 'sw':
                loc, offset = divmod(self._read_reg(args[1]) + args[2], 4)
                assert offset == 0
                data = Comp_RV32I.twos_comp_bin(self._read_reg(args[0]), 32)
                self.mem[loc] = data
            # BRANCH
            case 'beq':
                if self._read_reg(args[0]) == self._read_reg(args[1]):
                    self.pc += args[2]-4 # -4 to counteract self.pc += 4 at end
            case 'bne':
                if self._read_reg(args[0]) != self._read_reg(args[1]):
                    self.pc += args[2] - 4
            case 'blt':
                if self._read_reg(args[0]) < self._read_reg(args[1]):
                    self.pc += args[2] - 4
            case 'bge':
                if self._read_reg(args[0]) >= self._read_reg(args[1]):
                    self.pc += args[2] - 4
            case 'bltu':
                if self._read_reg(args[0], signed=False) < self._read_reg(args[1], signed=False):
                    self.pc += args[2] - 4
            case 'bgeu':
                if self._read_reg(args[0], signed=False) >= self._read_reg(args[1], signed=False):
                    self.pc += args[2] - 4
            # JUMP
            case 'jal':
                self._set_reg(args[0], self.pc + 4)
                self.pc += args[1] - 4
            case 'jalr':
                self._set_reg(args[0], self.pc + 4)
                self.pc += self._read_reg(args[1]) + args[2] - 4
            # LOAD UPPER
            case 'lui':
                self._set_reg(args[0], args[1])
            case 'auipc':
                self._set_reg(args[0], self.pc + args[1])
            # SYSTEM
            case 'ecall' | 'ebreak':
                # not properly implemented, just HALT
                self.status = 'HALT'
                if self.debug:
                    print("RISCV HALT")

        self.pc += 4
        return instr, *args