import pickle
import pandas as pd
from ..pre_processing import filter_NaNpeptides
from .common import subplotLabel, getSetup
from ..clustering import DDMC

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    _, _ = getSetup((15, 10), (3, 5))

    # Signaling
    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
    X = filter_NaNpeptides(X, tmt=2)
    d = X.select_dtypes(include=['float64']).T
    i = X.select_dtypes(include=['object'])

    model = DDMC(i, ncl=25, SeqWeight=5, distance_method="PAM250")
    model.fit(d, "NA", nRepeats=1)
    with open('msresist/data/pickled_models/pam250/CPTACmodel_pam250_CL20_W0', 'wb') as m:
        pickle.dump([model], m)

    print("model dumped.")
    raise SystemExit
