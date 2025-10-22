import pytest

from modules.vector import Vector
from modules.csr import CSR
from modules.RV32I import RV32I
from riscv import RISCV
from test_commons import *

tests = []

sim = RISCV([RV32I, CSR, Vector], 16, [], debug=True)
mod = sim.modules['Vector']


@pytest.mark.parametrize('instr, binary', tests)

def test_assemble(instr, binary):
    tcom_assemble(sim, mod, instr, binary)
@pytest.mark.parametrize('instr, binary', tests)


def test_unassemble(instr, binary):
    tcom_unassemble(mod, instr, binary)