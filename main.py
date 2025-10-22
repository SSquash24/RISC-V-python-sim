"""
main.py
Provides tkinter GUI for the sim
usage: run with no args for GUI file prompt, or give the file as first argument
"""

import sys
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter import messagebox as mb

from riscv import RISCV
from modules.RISC_modules import RVError, AssemblerError
from modules.RV32I import RV32I
from modules.csr import CSR
from modules.Float import Float
from modules.vector import Vector

def file_prompt():
    return fd.askopenfilename(title="Select code file")


class DisplaySim:

    def set_mem_list(self, begin):
        self.memList.delete(0, tk.END)
        for i in range(int(begin/4), int(begin/4)+16):
            if i == self.sim.state['mem_size']: break
            self.memList.insert(tk.END, f"{i*self.sim.state['word_size']:04x}:  {int(self.sim.state['mem'][i], 2):0{2*self.sim.state['word_size']}x}")

    def __init__(self, mem_size, code, modules):
        self.sim = RISCV(modules, mem_size, code, debug=True)

        # create tkinter window
        self.root = tk.Tk()
        self.root.title("RISC-V Sim")

        #tkinter vars
        self.status_var = tk.StringVar()
        self.prev_instr_var = tk.StringVar()
        self.next_instr_var = tk.StringVar()
        self.mem_addr_var = tk.StringVar() # StringVar instead of IntVar to allow empty string for typing benefits
        self.pc_var = tk.StringVar()

        self.mem_addr_var.trace('w', lambda *_: self.set_mem_list(int(self.mem_addr_var.get())) if self.mem_addr_var.get() else None)

        # infoFrame displays sim status, prev and next instruction
        infoFrame = ttk.LabelFrame(self.root, text="Info")

        ttk.Label(infoFrame, text="Status:").grid(column=0, row=0)
        ttk.Label(infoFrame, textvariable=self.status_var).grid(column=1, row=0)


        ttk.Label(infoFrame, text="Last Instruction:").grid(column=0, row=2)
        self.prev_instr_var.set("NULL")
        ttk.Label(infoFrame, textvariable=self.prev_instr_var).grid(column=1, row=2)

        ttk.Label(infoFrame, text="Next Instruction:").grid(column=0, row=3)
        ttk.Label(infoFrame, textvariable=self.next_instr_var).grid(column=1, row=3)

        infoFrame.grid(row=0, column=2)


        # runFrame contains controls to run/reset the sim
        runFrame = ttk.LabelFrame(self.root, text="Run")
        ttk.Button(runFrame, text="step", command=self.tk_step).grid(column=0, row=0)
        ttk.Button(runFrame, text="run", command=self.tk_run).grid(column=2, row=0)
        ttk.Button(runFrame, text="reset", command=self.tk_reset).grid(column=1, row=1)
        runFrame.grid(row=1, column=2)

        # memFrame lists the main memory of the sim
        memFrame = ttk.LabelFrame(self.root, text="Memory")
        self.memList = tk.Listbox(memFrame, selectmode=tk.BROWSE, height=16, font='TkFixedFont')
        self.memList.grid(column=0, row=0, columnspan=2)


        memCtrFrame = ttk.Frame(memFrame)
        ttk.Label(memCtrFrame, text="View Addr:").grid(column=0, row=0)
        ttk.Spinbox(memCtrFrame, textvariable=self.mem_addr_var, from_=0, to=(mem_size-1)*4, width=8, validate="key",
                    validatecommand=(self.root.register(lambda s: 0 <= int(s) < mem_size*4 if s.isdigit() else s == ''), '%P'),
                    increment=self.sim.state['word_size']).grid(column=1, row=0)
        memCtrFrame.grid(column=0, row=1)

        memFrame.grid(row=0, column=1, rowspan=2)

        # regFrame lists the register values (including pc and x0)
        regFrame = ttk.LabelFrame(self.root, text="Registers")
        ttk.Label(regFrame, text="PC:", width=3).grid(column=0, row=0, sticky="e")
        ttk.Label(regFrame, textvariable=self.pc_var, width=18).grid(column=1, row=0, sticky="w")


        self.reglist = tk.Listbox(regFrame, selectmode=tk.BROWSE, height=16, width=30, font='TkFixedFont')
        self.reglist.grid(column=0, row=1, columnspan=2)

        if Float in modules:
            self.freglist = tk.Listbox(regFrame, selectmode=tk.BROWSE, height=16, width=36, font='TkFixedFont')
            self.freglist.grid(column=2, row=1)
        else:
            self.freglist = None

        if Vector in modules:
            self.vreglist = tk.Listbox(regFrame, selectmode=tk.BROWSE, height=16, font='TkFixedFont')
            self.vreglist.grid(column=1, row=2, columnspan=2)
        else:
            self.vreglist = None

        regFrame.grid(column=0, row=0, rowspan=3)

        if CSR in modules:
            csrFrame = ttk.LabelFrame(self.root, text="CSR")
            self.csrList = tk.Listbox(csrFrame, selectmode=tk.BROWSE, height=16, width=32, font='TkFixedFont')
            self.csrList.pack()
            csrFrame.grid(row=2,column=2, columnspan=2)
        else:
            self.csrList = None


        self.mem_addr_var.set('0')
        self.update_vars()
        self.root.mainloop()

    # called when the run button is pressed
    def tk_run(self):
        while self.sim.state['status'] == 'RUNNING':
            try:
                instr = self.sim.step()
                if instr is not None:
                    self.prev_instr_var.set(instr)
            except RVError as e:
                mb.showerror("RISC-V Error", str(e))
                break
        self.update_vars()

    # called when the reset button is pressed
    def tk_reset(self):
        self.sim.reset()
        self.update_vars()

    # called when the step button is pressed
    def tk_step(self):
        try:
            instr = self.sim.step()
            if instr is not None:
                self.prev_instr_var.set(instr)
        except RVError as e:
            mb.showerror("RISC-V Error", str(e))
        self.update_vars()


    def update_vars(self):
        #regs
        self.pc_var.set(hex(self.sim.state['pc']))
        self.status_var.set(self.sim.state['status'])

        self.reglist.delete(0, tk.END)
        for i, reg in enumerate(self.sim.state['regs']):
            self.reglist.insert(tk.END, f"x{str(i):<3}: {reg:0{2*self.sim.state['word_size']}x} = {reg}")

        if self.freglist is not None:
            self.freglist.delete(0, tk.END)
            for i, reg in enumerate(self.sim.state['fregs']):
                self.freglist.insert(tk.END, f"f{str(i):<3}: {'%08X' % int(reg.bits(), 2)} = {reg.value()}")

        if self.vreglist is not None:
            self.vreglist.delete(0, tk.END)
            for i, reg in enumerate(self.sim.state['vregs']):
                self.vreglist.insert(tk.END, f"v{str(i):<3}: {hex(int(reg, 2))[2:].rjust(self.sim.modules['Vector'].vlen//8, '0')}")

        if self.csrList is not None:
            self.csrList.delete(0, tk.END)
            inv_csr = {v: k for k, v in self.sim.state['csr_dict'].items()}
            for i, csr in self.sim.state['csrs'].items():
                self.csrList.insert(tk.END, f"{i} ({inv_csr[i].ljust(6)}): {csr[0](False)}") #csr[0](False) means read without side effects

        # next instr
        try:
            self.next_instr_var.set(self.sim.unassemble(self.sim.state['mem'][self.sim.state['pc'] // 4])[0])
        except RVError:
            self.next_instr_var.set(f"Unknown")

        #mem list
        self.mem_addr_var.set(self.mem_addr_var.get()) # triggers trace effect


# main code reads file and creates instance of DisplaySim (the class above)
if __name__ == '__main__':
    if len(sys.argv) >= 2:
        if sys.argv[1] in ['-h', "--help"]:
            print("Usage: python main.py <codeFile>\nLeave <codeFile> empty for a prompt to give the file")
            exit(0)
        else:
            filename = sys.argv[1]
    else:
        filename = file_prompt()

    modules = [RV32I]
    if '-c' in sys.argv:
        print("Using module CSR")
        modules.append(CSR)
    if '-f' in sys.argv:
        modules.append(Float)
    if '-v' in sys.argv:
        modules.append(Vector)
    try:
        code = open(filename, 'r').read().split('\n')
    except FileNotFoundError | PermissionError as e:
        mb.showerror('Cannot open file', str(e))
        quit()

    DisplaySim(32, code, modules)