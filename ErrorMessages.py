#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

import sys


def CatchErrorException(message):
    print('ERROR. ' + message + '... EXIT')
    sys.exit(0)

def CatchWarningException(message):
    print('WARNING. ' + message + '...')
    pass
