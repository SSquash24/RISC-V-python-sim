# main.py: Provides tkinter GUI for the sim

import sys
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter import messagebox as mb

from riscv import RISCV

def filePrompt():
    return fd.askopenfilename(title="Select code file")


class DisplaySim:
    def __init__(self, code):
        self.sim = RISCV(code, debug=True)
        self.root = tk.Tk()
        self.root.title("RISC-V Sim")

        infoFrame = ttk.LabelFrame(self.root, text="Info")
        ttk.Label(infoFrame, text="last instruction:").grid(column=0, row=0)
        self.Tk_prevInstr = tk.StringVar()
        self.Tk_prevInstr.set("NULL")
        ttk.Label(infoFrame, textvariable=self.Tk_prevInstr).grid(column=1, row=0)
        infoFrame.pack(side=tk.TOP, fill=tk.X)

        runFrame = ttk.LabelFrame(self.root, text="Run")
        ttk.Button(runFrame, text="step", command=self.tk_step).grid(column=0, row=1)
        runFrame.pack(side=tk.TOP, fill=tk.X)

        self.root.mainloop()

    def tk_step(self):
        self.sim.step()
        self.updateVars()

    def updateVars(self):
        pass

if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] in ['-h', "--help"]:
            print("Usage: python main.py <codeFile>\nLeave <codeFile> empty for a prompt to give the file")
            exit(0)
        else:
            filename = sys.argv[1]
    else:
        filename = filePrompt()

    try:
        code = open(filename, mode='r').read().splitlines()
    except PermissionError:
        mb.showerror("SIM Error", f"Could not open file '{filename}'\nPermission denied.")
        exit(1)

    DisplaySim(code)