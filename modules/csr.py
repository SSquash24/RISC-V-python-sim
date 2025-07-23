from modules.RISC_modules import *
from yamls.read_yaml import read_yaml

class CSR(Module):

    opcodes, inv_opcodes = read_yaml('csr.yaml')

    def __init__(self, state):
        self.state = state
        state['csrs'] = {}  # csr is of form addr [bits] : (val [int], effect(read, write) [Bool, Bool -> None])


    def assemble(self, *instr):
        try:
            opcode, instr_type, *args = CSR.inv_opcodes[instr[0]]
        except KeyError:
            return False, None
        return True, self.assemble_I(opcode, self.reg(instr[1]), args[0], self.reg(instr[2]), int(instr[3]))

    def unassemble(self, binary):
        opcode, rd, rs1, _, funct3, _ = super()._decompile_commons(binary)
        try:
            _, _, table = CSR.opcodes[opcode]
            imm = self.twos_to_int(binary[0:12])
            return True, (table[funct3], rd, rs1, imm)
        except KeyError:
            return False, None

    def run_instr(self, instr):

        instr, *args = instr
        addr = self.twos_comp_bin(args[2], 12)
        if addr not in self.state['csrs'].keys():
            raise RVError(f"CSR Error: CSR {addr} does not exist.")
        writable = addr[:2] != '11'
        match instr:
            case  'csrrw':
                if not writable:
                    raise RVError(f"CSR Error: Cannot write to CSR: {instr[2]}")

                new_val = self._read_reg(args[1])
                self._set_reg(args[0], self.state['csrs'][addr][0])

                self.state['csrs'][addr][0] = new_val
                self.state['csrs'][addr][1](args[0] != 0, True) # call read/write effect
            case 'csrrs':
                if not writable and args[1] != 0:
                    raise RVError(f"CSR Error: Cannot write to CSR: {instr[2]}")

                new_val = self.state['csrs'][addr][0] | self._read_reg(args[1])
                self._set_reg(args[0], self.state['csrs'][addr][0])

                self.state['csrs'][addr][0] = new_val
                self.state['csrs'][addr][1](args[0] != 0, args[1] != 0)  # call read/write effect

            case 'csrrc':
                if not writable and args[1] != 0:
                    raise RVError(f"CSR Error: Cannot write to CSR: {instr[2]}")

                new_val = self.state['csrs'][addr][0] ^ ((1 << 32) - self._read_reg(args[1]) - 1)
                self._set_reg(args[0], self.state['csrs'][addr][0])

                self.state['csrs'][addr][0] = new_val
                self.state['csrs'][addr][1](args[0] != 0, args[1] != 0)  # call read/write effect

            case 'csrrwi':
                if not writable:
                    raise RVError(f"CSR Error: Cannot write to CSR: {instr[2]}")

                new_val = args[1]
                self._set_reg(args[0], self.state['csrs'][addr][0])

                self.state['csrs'][addr][0] = new_val
                self.state['csrs'][addr][1](args[0] != 0, True)  # call read/write effect

            case 'csrrsi':
                if not writable and args[1] != 0:
                    raise RVError(f"CSR Error: Cannot write to CSR: {instr[2]}")

                new_val = self.state['csrs'][addr][0] | args[1]
                self._set_reg(args[0], self.state['csrs'][addr][0])

                self.state['csrs'][addr][0] = new_val
                self.state['csrs'][addr][1](args[0] != 0, args[1] != 0)  # call read/write effect

            case 'csrrci':
                if not writable and args[1] != 0:
                    raise RVError(f"CSR Error: Cannot write to CSR: {instr[2]}")

                new_val = self.state['csrs'][addr][0] ^ ((1 << 32) - args[1] - 1)
                self._set_reg(args[0], self.state['csrs'][addr][0])

                self.state['csrs'][addr][0] = new_val
                self.state['csrs'][addr][1](args[0] != 0, args[1] != 0)  # call read/write effect