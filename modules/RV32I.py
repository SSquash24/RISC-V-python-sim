"""
RV32I.py
Implements base RV32I as a Module (as defined in RISC_modules.py)
"""


from modules.RISC_modules import *

class RV32I(Module):
    opcodes, inv_opcodes, pseudos = read_yaml('rv32i.yaml', ret_Pseudos=True)
    order = 0

    def __init__(self, state):
        super().__init__(state)
        state['regs'] = [0 for _ in range(32)]
        state['word_size'] = 4
        state['reg_count'] = 32
        state['pseudos'] |= self.pseudos

    # Module implementations
    def unassemble(self, binary):
        # assert instr_type in Comp_RV32I.base_instr_types
        assert len(binary) == 32

        opcode, rd, rs1, rs2, funct3, funct7 = super()._decompile_commons(binary)

        try:
            args1 = RV32I.opcodes[opcode]
        except KeyError:
            return False, None

        instr_type = args1[1]

        try:
            match instr_type:
                case 'R':
                    assert args1[-2] == [14, 12]
                    assert args1[-1][funct3][-2] == [31, 25]
                    return True, (args1[-1][funct3][-1][funct7], rd, rs1, rs2)
                case 'I':
                    instr = args1[-1][funct3]
                    imm = Module.twos_to_int(binary[0:12])
                    if type(instr) != str:
                        if instr[-2] == [31, 20]:
                            return True, (instr[-1][imm], rd, rs1)
                        assert instr[-2] == [31, 25]
                        instr = instr[1][funct7]
                        imm = int(binary[7:12], 2)
                    return True, (instr, rd, rs1, imm)
                case 'S':
                    imm = Module.twos_to_int(binary[:7] + binary[20:25])
                    return True, (args1[-1][funct3], rs1, rs2, imm)
                case 'B':
                    imm = Module.twos_to_int(binary[0] + binary[24] + binary[1:7] + binary[20:24] + '0')
                    return True, (args1[-1][funct3], rs1, rs2, imm)
                case 'U':
                    return True, (args1[0], rd, Module.twos_to_int(binary[:20]))
                case 'J':
                    imm = Module.twos_to_int(binary[0] + binary[12:20] + binary[11] + binary[1:11] + '0')
                    return True, (args1[0], rd, imm)
                case 'SYS':
                    assert args1[-2] == [19, 7]
                    assert rd == 0 and rs1 == 0 and funct3 == 0
                    args1 = args1[-1][0]
                    assert args1[-2] == [31, 20]
                    imm = int(binary[0:12])
                    return True, (args1[-1][imm],)
                case _:
                    return False, None
        except (KeyError, AssertionError):
            pass
        return False, None

    def assemble(self, *instr):

        try:
            opcode, instr_type, args = RV32I.inv_opcodes[instr[0]]
        except KeyError:
            return False, None
        match instr_type:
            case 'R':
                return True, RV32I.assemble_32(opcode, rd=Module.reg(instr[1]), rs1=Module.reg(instr[2]), rs2=Module.reg(instr[3]), flags=args)
            case 'I':
                if len(instr) == 4:
                    return True, Module.assemble_I(opcode, instr[1], instr[2], int(instr[3]), args)
                else:
                    return True, Module.assemble_32(opcode, rd=Module.reg(instr[1]), rs1=Module.reg(instr[2]), flags=args)
            case 'S':
                return True, RV32I.assemble_S(opcode, instr[1], instr[2], int(instr[3]), flags=args)
            case 'B':
                return True, RV32I.assemble_B(opcode, instr[1], instr[2], int(instr[3]), flags=args)
            case 'U':
                return True, RV32I.assemble_U(opcode, instr[1], int(instr[2]))
            case 'J':
                return True, RV32I.assemble_J(opcode, instr[1], int(instr[2]))
            case 'SYS':
                return True, RV32I.assemble_32(opcode, flags=args)
            case _:
                return False, None

    def run_instr(self, instr):

        # helper func to convert from unsigned to signed
        def revert(num, size=32):
            return num if num < (1 << size - 1) else (1 << size) - num

        instr, *args = instr
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
                self._set_reg(args[0], self.state['regs'][args[1]] >> self._read_reg(args[2]))
            case 'slt':
                self._set_reg(args[0], 1 if self.state['regs'][args[1]] < self._read_reg(args[2]) else 0)
            case 'sltu':
                self._set_reg(args[0],
                              1 if self._read_reg(args[1], signed=False) < self._read_reg(args[2], signed=False) else 0)
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
                self._set_reg(args[0],
                              revert((Module.twos_to_int(self.state['mem'][loc]) >> 8 * offset) % (1 << 8), size=8))
            case 'lh':
                loc, offset = divmod(self._read_reg(args[1]) + args[2], 4)
                assert loc % 2 == 0
                self._set_reg(args[0],
                              revert((Module.twos_to_int(self.state['mem'][loc]) >> 8 * offset) % (1 << 16), size=16))
            case 'lw':
                loc, offset = divmod(self._read_reg(args[1]) + args[2], 4)
                assert offset == 0
                self._set_reg(args[0], Module.twos_to_int(self.state['mem'][loc]))
            case 'lbu':
                loc, offset = divmod(self._read_reg(args[1]) + args[2], 4)
                self._set_reg(args[0], (int(self.state['mem'][loc], 2) >> 8 * offset) % (1 << 8))
            case 'lhu':
                loc, offset = divmod(self._read_reg(args[1]) + args[2], 4)
                assert loc % 2 == 0
                self._set_reg(args[0], revert((int(self.state['mem'][loc], 2) >> 8 * offset) % (1 << 16)))
            # STORE
            case 'sb':
                loc, offset = divmod(self._read_reg(args[1]) + args[2], 4)
                data = bin(self._read_reg(args[0], signed=False) % (1 << 8))[2:]
                self.state['mem'][loc] = self.state['mem'][loc][:(3 - offset) * 8] + data + self.state['mem'][loc][
                                                                                            :(offset + 1) * 8]
            case 'sh':
                loc, offset = divmod(self._read_reg(args[1]) + args[2], 4)
                assert offset % 2 == 0
                data = bin(self._read_reg(args[0], signed=False) % (1 << 16))[2:]
                self.state['mem'][loc] = self.state['mem'][loc][:(1 - offset) * 16] + data + self.state['mem'][loc][
                                                                                             :(offset + 1) * 16]
            case 'sw':
                loc, offset = divmod(self._read_reg(args[1]) + args[2], 4)
                assert offset == 0
                data = Module.twos_comp_bin(self._read_reg(args[0]), 32)
                self.state['mem'][loc] = data
            # BRANCH
            case 'beq':
                if self._read_reg(args[0]) == self._read_reg(args[1]):
                    self.state['pc'] += args[2] - 4  # -4 to counteract self.state['pc'] += 4 at end
            case 'bne':
                if self._read_reg(args[0]) != self._read_reg(args[1]):
                    self.state['pc'] += args[2] - 4
            case 'blt':
                if self._read_reg(args[0]) < self._read_reg(args[1]):
                    self.state['pc'] += args[2] - 4
            case 'bge':
                if self._read_reg(args[0]) >= self._read_reg(args[1]):
                    self.state['pc'] += args[2] - 4
            case 'bltu':
                if self._read_reg(args[0], signed=False) < self._read_reg(args[1], signed=False):
                    self.state['pc'] += args[2] - 4
            case 'bgeu':
                if self._read_reg(args[0], signed=False) >= self._read_reg(args[1], signed=False):
                    self.state['pc'] += args[2] - 4
            # JUMP
            case 'jal':
                self._set_reg(args[0], self.state['pc'] + 4)
                self.state['pc'] += args[1] - 4
            case 'jalr':
                self._set_reg(args[0], self.state['pc'] + 4)
                self.state['pc'] += self._read_reg(args[1]) + args[2] - 4
            # LOAD UPPER
            case 'lui':
                self._set_reg(args[0], args[1])
            case 'auipc':
                self._set_reg(args[0], self.state['pc'] + args[1])
            # SYSTEM
            case 'ecall' | 'ebreak':
                # not properly implemented, HALT
                self.state['status'] = 'HALT'
                if self.state['debug']:
                    print("RISCV HALT")

    def reset_module(self):
        self.state['regs'] = [0 for _ in range(self.state['reg_count'])]


