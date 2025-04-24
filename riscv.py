class RISCV:

    def __init__(self, debug=False, wordlength=32):
        assert(wordlength in [32,64])
        assert(type(debug) is bool)
        self.debug = debug
        self.wordlength = wordlength

    def runcommand(self, command):
        assert(type(command) is str)
        if self.debug:
            print("RISCV command: " + command)

