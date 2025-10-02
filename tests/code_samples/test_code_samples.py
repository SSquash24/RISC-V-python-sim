import pytest, os
from modules.RV32I import RV32I
from modules.Float import Float
from modules.csr import CSR
from riscv import RISCV

EXTENSION = '.txt'
END_SPLITTER = 'RESULTS---'
MODULES = 'MODULES:'
MEM = 'MEM:'

modules_dict = {
    CSR.__name__: CSR,
    Float.__name__: Float,
    RV32I.__name__: RV32I
}

tests = []
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(EXTENSION):
        tests.append(file)


@pytest.mark.parametrize('filename', tests)
def test_code_samples(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), 'r') as f:
        content = f.read().split('\n')
        modules = content[0]
        assert modules.startswith(MODULES)
        modules = modules.split(' ')[1:]
        modules = [modules_dict[m] for m in modules]

        mem = content[1]
        assert mem.startswith(MEM)
        mem = int(mem.split(' ')[1])

        content = content[2:]
        split = content.index(END_SPLITTER)
        code = content[:split]
        res = content[split+1:]

        sim = RISCV(modules=modules, mem_size=mem, code=code)
        sim.run()

        #test regs match expected results
        for check in res:
            reg, val = check.split('=')
            match reg[0]:
                case 'x':
                    #normal reg
                    reg = int(reg[1:])
                    assert sim.state['regs'][reg] == int(val)
                case 'f':
                    #float reg
                    reg = int(reg[1:])
                    assert sim.state['fregs'][reg].value() == float(val)
                case _:
                    assert False

