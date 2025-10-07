def tcom_assemble(sim, module, instr, binary):
    success, bits = module.assemble(*sim.clean_instr(instr))
    assert success
    assert bits == binary

def tcom_unassemble(module, instr, binary):
    success, packed = module.unassemble(binary)
    assert success
    instr = instr.replace(',', '').split()

    def clean(x: str):
        try:
            return int(x)
        except ValueError:
            if x[0] in ['x', 'f'] and x[1:].isdigit():
                return module.reg(x)
            else:
                return x

    instr = tuple(clean(i) for i in instr)
    assert packed == instr