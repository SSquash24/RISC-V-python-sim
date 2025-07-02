# main.py: Provides tkinter GUI for the sim

import sys
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter import messagebox as mb

from compiler import Comp_RV32I
from riscv import RV32I

def filePrompt():
    return fd.askopenfilename(title="Select code file")


class DisplaySim:

    def setMemList(self, begin):
        self.memList.delete(0, tk.END)
        for i in range(begin, begin+16):
            if i == self.sim.mem_size: break
            self.memList.insert(tk.END, f"{str(i).zfill(3)}:  {hex(self.sim.mem[i])}")

    def __init__(self, mem_size, code):
        self.sim = RV32I(mem_size, code, debug=True)
        self.root = tk.Tk()
        self.root.title("RISC-V Sim")

        infoFrame = ttk.LabelFrame(self.root, text="Info")
        ttk.Label(infoFrame, text="last instruction:").grid(column=0, row=0)
        self.Tk_prevInstr = tk.StringVar()
        self.Tk_prevInstr.set("NULL")
        ttk.Label(infoFrame, textvariable=self.Tk_prevInstr).grid(column=1, row=0)
        infoFrame.grid(row=0, column=2)

        runFrame = ttk.LabelFrame(self.root, text="Run")
        ttk.Button(runFrame, text="step", command=self.tk_step).grid(column=0, row=1)
        runFrame.grid(row=1, column=2)

        memFrame = ttk.LabelFrame(self.root, text="Memory")
        self.memList = tk.Listbox(memFrame, selectmode=tk.BROWSE, height=16, font='TkFixedFont')
        self.memList.grid(column=0, row=0, columnspan=2)


        memCtrFrame = ttk.Frame(memFrame)
        ttk.Label(memCtrFrame, text="View Addr:").grid(column=0, row=0)
        self.mem_addr_var = tk.StringVar() # StringVar instead of IntVar to allow empty string for typing benefits
        self.mem_addr_var.trace('w', lambda *_: self.setMemList(int(self.mem_addr_var.get())) if self.mem_addr_var.get() else None)
        ttk.Spinbox(memCtrFrame, textvariable=self.mem_addr_var, from_=0, to=mem_size-1, width=8, validate="key",
                    validatecommand=(self.root.register(lambda s: 0 <= int(s) < mem_size if s.isdigit() else s == ''), '%P')
                    ).grid(column=1, row=0)
        memCtrFrame.grid(column=0, row=1)

        self.mem_addr_var.set('0')
        memFrame.grid(row=0, column=1, rowspan=2)

        regFrame = ttk.LabelFrame(self.root, text="Registers")
        self.pc_var = tk.StringVar()
        ttk.Label(regFrame, text="PC:", width=3).grid(column=0, row=0, sticky="e")
        ttk.Label(regFrame, textvariable=self.pc_var, width=18).grid(column=1, row=0, sticky="w")

        self.reglist = tk.Listbox(regFrame, selectmode=tk.BROWSE, height=16, font='TkFixedFont')
        self.reglist.grid(column=0, row=1, columnspan=2)

        regFrame.grid(column=0, row=0, rowspan=2)

        self.updateVars()
        self.root.mainloop()

    def tk_step(self):
        self.sim.step()
        self.updateVars()

    def updateVars(self):
        #regs
        self.pc_var.set(hex(self.sim.pc))

        self.reglist.delete(0, tk.END)
        for i, reg in enumerate(self.sim.regs):
            self.reglist.insert(tk.END, f"{'x'+str(i):>3}: {hex(reg)}")

        pass

if __name__ == '__main__':
    # if len(sys.argv) == 2:
    #     if sys.argv[1] in ['-h', "--help"]:
    #         print("Usage: python main.py <codeFile>\nLeave <codeFile> empty for a prompt to give the file")
    #         exit(0)
    #     else:
    #         filename = sys.argv[1]
    # else:
    #     filename = filePrompt()
    #
    # try:
    #     code = open(filename, mode='r').read().splitlines()
    # except PermissionError:
    #     mb.showerror("SIM Error", f"Could not open file '{filename}'\nPermission denied.")
    #     exit(1)
    #
    # DisplaySim(32, code)
    DisplaySim(32, [i for i in range(16)])