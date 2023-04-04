import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import string
import re
import datetime
import os

from cleantext import clean
import contractions

from sklearn.preprocessing import MinMaxScaler, StandardScaler


def oneTimeSetup():
    print("Setup~")
    plt.rcParams["figure.figsize"] = (20,10)

