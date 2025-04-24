import sys
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter import messagebox as mb

from riscv import RISCV

def filePrompt():
    return fd.askopenfilename(title="Select code file")


def display_sim(code):
    sim = RISCV(debug=True)
    for i in code:
        sim.runcommand(i)

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

    display_sim(code)