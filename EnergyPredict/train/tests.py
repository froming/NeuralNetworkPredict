from django.test import TestCase

import datetime
import random

# Create your tests here.
def getImageName():
    filenameOne = str(datetime.datetime.now()).split(' ')[0]
    filenameTwo = str(random.random()).split('.')[1]
    filename = filenameOne + '_' + filenameTwo
    return filename