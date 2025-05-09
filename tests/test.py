#%%
import os
import pickle 
import random
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path 
from MASS.src.tools import spice
from MASS.src.tools.core import SAMPLE 
from MASS.src.tools.core import GROUP
# %%
dir = Path("/scratcha/stlab-icgc/users/somers01/CNProj/Slowmalier/Testing/SomeSamples/")

assignment_sheet = Path("/home/somers01/CNProj/Slowmalier/Data/sample_assignment.csv")
assignment_df = pd.read_csv(assignment_sheet, sep=",", names = ["sample_ID", "group_ID"])
sample_dict = assignment_df.groupby('group_ID')['sample_ID'].apply(list).to_dict()
group_IDs = sample_dict.keys()
workable_groups, locked_paths = spice.SearchGroups(sample_dict, group_IDs, dir)


# %%
