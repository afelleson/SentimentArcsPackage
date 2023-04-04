import sys
import pytest
from imppkg.hello import doit
from imppkg.simplifiedSA import oneTimeSetup


def test_always_pass():
    doit()
    oneTimeSetup()
    assert True