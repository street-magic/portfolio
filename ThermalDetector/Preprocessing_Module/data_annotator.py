import matplotlib.pyplot as plt
from matplotlib import cm as _pltcolormap
from matplotlib import colors as _pltcolors
import numpy as np
import pandas as pd
import os
import sys
import csv
import pathlib
import sys

dirname = sys.argv[1]

for file in (pathlib.Path().cwd() / dirname).rglob('*.csv'):
    #df = pd.from_csv(str(file))
    #df.to_csv(str(file.with_name(file.stem + '.csv')), sep='\t')
    par_name = file.parent.name
    NOP = par_name[par_name.find('NOP-')+len('NOP-')]
    bed_present = '1' if 'BED' in par_name else '0'
    bed_occupied = '1' if 'laying' in par_name else '0'
    print(par_name + "  ~~  " + NOP + ',' + bed_present + ',' + bed_occupied)

    df = pd.read_csv(str(file))
    df["NOP"] = NOP
    df["bed_present"] = bed_present
    df["bed_occupied"] = bed_occupied
    df.to_csv(file.absolute(), index=False)
