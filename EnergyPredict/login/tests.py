from django.test import TestCase
from numpy import byte
import hashlib

# Create your tests here.
def EnCookie(username,password):
    BeforeEnCookieStr = str(username) + str(password)
    BeforeEnCookieByte = bytes(BeforeEnCookieStr.encode('UTF-8'))
    AfterEnCookieByte = hashlib.md5(BeforeEnCookieByte)
    AfterEnCookieStr = AfterEnCookieByte.hexdigest()
    return str(AfterEnCookieStr) + '='