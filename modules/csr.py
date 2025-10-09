from modules.RISC_modules import *
from yamls.read_yaml import read_yaml

class CSR(Module):

    opcodes, inv_opcodes = read_yaml('csr.yaml')
    order = 1

    """ Set csr with side effects"""
    def set_csr(self, a, val):
        if type(a) is int:
            a = bin(a)[2:].rjust(32, '0')

        old_val = self.state['csr'][a][0]
        self.state['csr'][a][0] = self.state['csr'][a][1](old_val, val, False, True)
        return self.state['csr'][a][0]

    def __init__(self, state):
        super().__init__(state)
        state['csrs'] = {}  # csr is of form addr [bits] : (read_func (Bool = True -> int), write_func (int -> None))
        state['set_csr'] = lambda a, x: self.set_csr(a, x) # access to set_csr func for other modules

    def assemble(self, *instr):
        try:
            opcode, instr_type, *args = CSR.inv_opcodes[instr[0]]
        except KeyError:
            return False, None
        assert instr_type == 'I'
        if instr[0][-1] == 'i':
            return True, self.assemble_I(opcode, self.reg(instr[1]), int(instr[2]), int(instr[3]), args[0])
        else:
            return True, self.assemble_I(opcode, self.reg(instr[1]), self.reg(instr[2]), int(instr[3]), args[0])

    def unassemble(self, binary):
        opcode, rd, rs1, _, funct3, _ = super()._decompile_commons(binary)
        try:
            _, _, _, table = CSR.opcodes[opcode]
            imm = self.twos_to_int(binary[0:12])
            return True, (table[funct3], rd, rs1, imm)
        except (KeyError, ValueError):
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
                if args[0] != 0:
                    old_val = self.state['csrs'][addr][0]() #read
                    self._set_reg(args[0], old_val)


                self.state['csrs'][addr][1](new_val) #write value
            case 'csrrs':
                if not writable and args[1] != 0:
                    raise RVError(f"CSR Error: Cannot write to CSR: {instr[2]}")

                old_val = self.state['csrs'][addr][0](args[0] != 0) #read
                new_val = old_val | self._read_reg(args[1])
                self._set_reg(args[0], old_val)

                if args[1] != 0:
                    self.state['csrs'][addr][1](new_val)  # write value

            case 'csrrc':
                if not writable and args[1] != 0:
                    raise RVError(f"CSR Error: Cannot write to CSR: {instr[2]}")

                old_val = self.state['csrs'][addr][0](args[0] != 0) # read
                new_val = old_val ^ ((1 << 32) - self._read_reg(args[1]) - 1)
                self._set_reg(args[0], old_val)

                if args[1] != 0:
                    self.state['csrs'][addr][1](new_val)  # write val

            case 'csrrwi':
                if not writable:
                    raise RVError(f"CSR Error: Cannot write to CSR: {instr[2]}")
                new_val = args[1]
                if args[0] != 0:
                    old_val = self.state['csrs'][addr][0]() # read
                    self._set_reg(args[0], old_val)

                self.state['csrs'][addr][1](new_val)  # write val

            case 'csrrsi':
                if not writable and args[1] != 0:
                    raise RVError(f"CSR Error: Cannot write to CSR: {instr[2]}")

                old_val = self.state['csrs'][addr][0](args[0] != 0)
                new_val = old_val | args[1]
                self._set_reg(args[0], old_val)

                if args[1] != 0:
                    self.state['csrs'][addr][1](new_val)  # call read/write effect

            case 'csrrci':
                if not writable and args[1] != 0:
                    raise RVError(f"CSR Error: Cannot write to CSR: {instr[2]}")

                old_val = self.state['csrs'][addr][0](args[0] != 0)
                new_val = old_val ^ ((1 << 32) - args[1] - 1)
                self._set_reg(args[0], old_val)

                if args[1] != 0:
                    self.state['csrs'][addr][1](new_val)  # call read/write effect