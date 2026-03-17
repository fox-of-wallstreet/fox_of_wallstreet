'''
Place for global things.
'''
import os
import sys

def fnline():
    '''
    For logging and tracing.
    Returns current filename and line number.
    E.g.: backtest.py(144))
    '''
    return os.path.basename(sys.argv[0]) + '(' + str(sys._getframe(1).f_lineno) + '):'
