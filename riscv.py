"""
riscv.py
Implements the simulator as a class that handles many 'Modules'
Modules are defined in the modules folder, with layout as defined in RISC_modules.py
"""

from modules.RISC_modules import RVError, Module
from code_cleaner import clean_code





class RISCV:


    def __init__(self, modules, mem_size, code, debug=False):
        self.state = {'mem_size': mem_size, 'modules':[m.__name__ for m in modules], 'pc': 0, 'mem': [], 'debug': debug, 'status': 'RUNNING', 'pseudos': {}}
        self.modules = {}
        modules.sort(key=lambda m: m.order)
        for m in modules:
            self.modules[m.__name__] = m(self.state)

        self.unassembled_code, self.data = clean_code(code, self.state['pseudos'], self.state['word_size'])

        self.code = [self.assemble(*instr)[0] for instr in self.unassembled_code] + self.data

        self.state['mem'] = self.code + ['1' * 8 * self.state['word_size']] * (self.state['mem_size'] - len(self.code))
        for m in self.modules.values():
            m.reset_module()

    def reset(self):
        if self.state['debug']:
            print("Resetting RISCV simulator")
        self.state['pc'] = 0
        self.state['mem'] = self.code + ['1' * 8 * self.state['word_size']] * (self.state['mem_size'] - len(self.code))
        self.state['status'] = 'RUNNING'
        for m in self.modules.values():
            m.reset_module()


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