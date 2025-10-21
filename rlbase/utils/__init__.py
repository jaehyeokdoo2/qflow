# This wraps the Rlbase package in a debugging wrapper.
import sys
import os
import pdb
import traceback
def debughook(etype, value, tb):
    traceback.print_exception(etype, value, tb)
    # if keyboardinterrup, just quite.
    if not issubclass(etype, KeyboardInterrupt):
        print() # make a new line before launching post-mortem
        pdb.pm() # post-mortem debugger
    os._exit(0)
sys.excepthook = debughook

# Ignore warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 