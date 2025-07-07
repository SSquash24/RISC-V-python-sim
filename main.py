# main.py: Provides tkinter GUI for the sim

import sys
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter import messagebox as mb

from compiler import Comp_RV32I, RVError
from riscv import RV32I

def file_prompt():
    return fd.askopenfilename(title="Select code file")


class DisplaySim:

    def set_mem_list(self, begin):
        self.memList.delete(0, tk.END)
        for i in range(int(begin/4), int(begin/4)+16):
            if i == self.sim.mem_size: break
            self.memList.insert(tk.END, f"{i*self.sim.word_size:04x}:  {int(self.sim.mem[i], 2):0{2*self.sim.word_size}x}")

    def __init__(self, mem_size, code):
        self.sim = RV32I(mem_size, code, debug=True)

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


        infoFrame = ttk.LabelFrame(self.root, text="Info")

        ttk.Label(infoFrame, text="Status:").grid(column=0, row=0)
        ttk.Label(infoFrame, textvariable=self.status_var).grid(column=1, row=0)


        ttk.Label(infoFrame, text="Last Instruction:").grid(column=0, row=2)
        self.prev_instr_var.set("NULL")
        ttk.Label(infoFrame, textvariable=self.prev_instr_var).grid(column=1, row=2)

        ttk.Label(infoFrame, text="Next Instruction:").grid(column=0, row=3)
        ttk.Label(infoFrame, textvariable=self.next_instr_var).grid(column=1, row=3)

        infoFrame.grid(row=0, column=2)



        runFrame = ttk.LabelFrame(self.root, text="Run")
        ttk.Button(runFrame, text="step", command=self.tk_step).grid(column=0, row=1)
        runFrame.grid(row=1, column=2)

        memFrame = ttk.LabelFrame(self.root, text="Memory")
        self.memList = tk.Listbox(memFrame, selectmode=tk.BROWSE, height=16, font='TkFixedFont')
        self.memList.grid(column=0, row=0, columnspan=2)


        memCtrFrame = ttk.Frame(memFrame)
        ttk.Label(memCtrFrame, text="View Addr:").grid(column=0, row=0)
        ttk.Spinbox(memCtrFrame, textvariable=self.mem_addr_var, from_=0, to=(mem_size-1)*4, width=8, validate="key",
                    validatecommand=(self.root.register(lambda s: 0 <= int(s) < mem_size*4 if s.isdigit() else s == ''), '%P'),
                    increment=self.sim.word_size).grid(column=1, row=0)
        memCtrFrame.grid(column=0, row=1)

        memFrame.grid(row=0, column=1, rowspan=2)

        regFrame = ttk.LabelFrame(self.root, text="Registers")
        ttk.Label(regFrame, text="PC:", width=3).grid(column=0, row=0, sticky="e")
        ttk.Label(regFrame, textvariable=self.pc_var, width=18).grid(column=1, row=0, sticky="w")

        self.reglist = tk.Listbox(regFrame, selectmode=tk.BROWSE, height=16, font='TkFixedFont')
        self.reglist.grid(column=0, row=1, columnspan=2)

        regFrame.grid(column=0, row=0, rowspan=2)

        self.mem_addr_var.set('0')
        self.update_vars()
        self.root.mainloop()

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
        self.pc_var.set(hex(self.sim.pc))
        self.status_var.set(self.sim.status)

        self.reglist.delete(0, tk.END)
        for i, reg in enumerate(self.sim.regs):
            self.reglist.insert(tk.END, f"{'x'+str(i):>3}: {reg:0{2*self.sim.word_size}x}")

        #next instr
        try:
            self.next_instr_var.set(Comp_RV32I.decompile_instr(self.sim.mem[self.sim.pc//4]))
        except RVError:
            self.next_instr_var.set(f"Unknown")

        #mem list
        self.mem_addr_var.set(self.mem_addr_var.get()) # triggers trace effect

if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] in ['-h', "--help"]:
            print("Usage: python main.py <codeFile>\nLeave <codeFile> empty for a prompt to give the file")
            exit(0)
        else:
            filename = sys.argv[1]
    else:
        filename = file_prompt()

    try:
        compiler = Comp_RV32I()
        code = compiler.compile_file(filename)
    except PermissionError:
        mb.showerror("SIM Error", f"Could not open file '{filename}'\nPermission denied.")
        exit(1)

    DisplaySim(32, code)