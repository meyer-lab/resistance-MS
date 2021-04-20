"""
This creates Figure 4: STK11 analysis
"""
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from .figure2 import plotDistanceToUpstreamKinase
from .figureM2 import find_patients_with_NATandTumor, merge_binary_vectors
from .figureM3 import plot_clusters_binaryfeatures, build_pval_matrix, calculate_mannW_pvals
from ..logistic_regression import plotROC, plotClusterCoefficients
from .common import subplotLabel, getSetup


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12, 7), (2, 3), multz={0: 1, 4: 1})

    # Add subplot labels
    subplotLabel(ax)

    # Phosphoproteomic aberrations associated with molecular signatures
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Load Clustering Model from Figure 2
    with open('msresist/data/pickled_models/binomial/CPTACmodel_BINOMIAL_CL24_W15_TMT2', 'rb') as p:
        model = pickle.load(p)[0]

    # Import Genotype data
    mutations = pd.read_csv("msresist/data/MS/CPTAC/Patient_Mutations.csv")
    mOI = mutations[["Sample.ID"] + list(mutations.columns)[45:54] + list(mutations.columns)[61:64]]
    y = mOI[~mOI["Sample.ID"].str.contains("IR")]

    # Find centers
    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
    centers = pd.DataFrame(model.transform())
    centers.columns = np.arange(model.ncl) + 1
    centers["Patient_ID"] = X.columns[4:]
    centers = centers.set_index("Patient_ID")

    # Hypothesis Testing
    assert np.all(y['Sample.ID'] == centers.index)
    centers["STK11"] = y["STK11.mutation.status"].values
    pvals = calculate_mannW_pvals(centers, "STK11", 1, 0)
    pvals = build_pval_matrix(model.ncl, pvals)
    plot_clusters_binaryfeatures(centers, "STK11", ["WT", "Mutant"], ax[0], pvals=pvals)
    ax[0].legend(loc='lower left')

    # Reshape data (Patients vs NAT and tumor sample per cluster)
    centers = centers.reset_index().set_index("STK11")
    centers = find_patients_with_NATandTumor(centers, "Patient_ID", conc=True)
    y = find_patients_with_NATandTumor(y.copy(), "Sample.ID", conc=False)
    assert all(centers.index.values == y.index.values), "Samples don't match"

    # Normalize
    centers = centers.T
    centers.iloc[:, :] = StandardScaler(with_std=False).fit_transform(centers.iloc[:, :])
    centers = centers.T

    # Logistic Regression
    centers["STK11"] = y["STK11.mutation.status"].values
    lr = LogisticRegressionCV(Cs=10, cv=10, solver="saga", max_iter=100000, tol=1e-4, n_jobs=-1, penalty="l1", class_weight="balanced")
    plotROC(ax[1], lr, centers.iloc[:, :-1].values, centers["STK11"], cv_folds=4, title="ROC STK11")
    plotClusterCoefficients(ax[2], lr.fit(centers.iloc[:, :-1], centers["STK11"].values), list(centers.columns[:-1]), title="STK11")
    ax[2].legend(loc='lower right', prop={'size': 8})

    # plot Upstream Kinases
    plotDistanceToUpstreamKinase(model, [1, 8, 9, 14, 22], ax[3], num_hits=3)

    return f
