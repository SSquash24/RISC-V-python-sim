"""
riscv.py
Implements the simulator as a class that handles many 'Modules'
Modules are defined in the modules folder, with layout as defined in RISC_modules.py
"""
import struct

from modules.RISC_modules import RVError, Module

class RISCV:


    def __init__(self, modules, mem_size, code, debug=False):
        self.state = {'mem_size': mem_size, 'modules':[m.__name__ for m in modules], 'pc': 0, 'mem': [], 'debug': debug, 'status': 'RUNNING', 'pseudos': {}}
        self.unassembled_code = code
        self.modules = {}
        modules.sort(key=lambda m: m.order)
        for m in modules:
            self.modules[m.__name__] = m(self.state)

        self.code = []
        for line in code:
            line = self.clean_instr(line)
            if not line:
                continue
            elif line[0][0] == '.':
                match line[0][1:]:
                    case 'int':
                        val = int(line[1])
                        self.code.append(Module.twos_comp_bin(val, self.state['word_size']*8))
                    case 'bool':
                        val = int(line[1])
                        self.code.append(Module.twos_comp_bin(val, 8).rjust(self.state['word_size']*8, '0'))
                    case 'float':
                        val = float(line[1])
                        bits, = struct.unpack('!I', struct.pack('!f', val))
                        self.code.append("{:032b}".format(bits))
            else:
                self.code.append(self.assemble(*line)[0])

        self.state['mem'] = self.code + ['0' * 8 * self.state['word_size']] * (self.state['mem_size'] - len(self.code))

    def reset(self):
        if self.state['debug']:
            print("Resetting RISCV simulator")
        self.state['pc'] = 0
        self.state['mem'] = self.code + ['0' * 8 * self.state['word_size']] * (self.state['mem_size'] - len(self.code))
        self.state['status'] = 'RUNNING'
        for m in self.modules.values():
            m.reset_module()

    def clean_instr(self, line):
        instr = line.strip().replace(',', '').split('#', 1)[0].split()
        if not instr:
            return []
        if instr[0] in self.state['pseudos'].keys():
            args = instr[1:]
            instr = self.state['pseudos'][instr[0]].copy()
            for i, arg in enumerate(instr):
                if arg.split(':')[0] == 'INP':
                    index = int(arg.split(':')[1])-1
                    instr[i] = args[index]
        return instr

    def assemble(self, instr, *args):
        for m in self.modules.values():
            success, res = m.assemble(instr, *args)
            if success:
                return res, m
        raise RVError(f'Failed to assemble RISCV line: {instr}, {args}')

    def unassemble(self, binary):
        for m in self.modules.values():
            success, res = m.unassemble(binary)
            if success:
                return res, m
        raise RVError(f'Failed to unassemble RISCV binary: {binary}')

    def step(self):
        binary = self.state['mem'][self.state['pc'] // self.state['word_size']]
        instr, module = self.unassemble(binary)
        if self.state['debug']:
            print(f"Step: Instr = {instr}. Handled by module '{type(module).__name__}'")
        module.run_instr(instr)
        self.state['pc'] += self.state['word_size']
        return instr

    # steps until status is no longer 'RUNNING'
    def run(self):
        while self.state['status'] == 'RUNNING':
            self.step()