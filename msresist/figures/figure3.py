"""
This creates Figure 3: ABL/SFK/YAP experimental validations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .common import subplotLabel, getSetup
from .figure1 import TimePointFoldChange, plot_IdSites
from msresist.pre_processing import preprocessing


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((16, 11), (4, 4), multz={4: 1, 6: 1, 8: 2, 13: 1})

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Import Dasatinib DR MS data
    X = preprocessing(AXL_Das_DR=True, Vfilter=True, log2T=True, mc_row=False)
    for i in range(X.shape[0]):
        X.iloc[i, 6:11] -= X.iloc[i, 6]
        X.iloc[i, 11:] -= X.iloc[i, 11]

    # Das DR time point
    plot_DasDR_timepoint(ax[0], "Dasatinib", "nM", time=96))

    # Luminex p-ASY Das DR
    plot_pAblSrcYap(ax[1:4])

    # Das DR Mass Spec Dose response cluster
    ax[4].axis("off")

    # AXL Mass Spec Cluster 4 enrichment of peptides in Das DR cluster
    plotHyerGeomTestDasDRGenes(ax[5])

    # Selected peptides within Dasatinib DR Cluster
    abl_sfk = {'LYN': 'Y397-p', 'YES1': 'Y223-p', 'ABL1': 'Y393-p', 'FRK': 'Y497-p', 'LCK': 'Y394-p'}
    plot_IdSites(ax[6], X, abl_sfk, "ABL&SFK", rn=False, ylim=False, xlabels=list(X.columns[6:]))

    # Das DR Mass Spec WT/KO dif
    ax[7].axis("off")

    return f


def plot_YAPinhibitorTimeLapse(ax, X, ylim=False):
    lines = ["WT", "KO"]
    treatments = ["UT", "E", "E/R", "E/A"]
    for i, line in enumerate(lines):
        for j, treatment in enumerate(treatments):
            if i > 0:
                j += 4
            m = X[X["Lines"] == line]
            m = m[m["Condition"] == treatment]
            sns.lineplot(x="Elapsed", y="Fold-change confluency", hue="Inh_concentration", data=m, ci=68, ax=ax[j])
            ax[j].set_title(line + "-" + treatment)
            if ylim:
                ax[j].set_ylim(ylim)
            if i != 0 or j != 0:
                ax[j].get_legend().remove()
            else:
                ax[j].legend(prop={'size': 10})


def transform_YAPviability_data(data, inhibitor, units, itp=24):
    """Transform to initial time point and convert into seaborn format"""
    new = []
    for i, mat in enumerate(data):
        if i > 0:
            mat = MeanTRs(mat)
        new.append(TimePointFoldChange(mat, itp))

    c = pd.concat(new, axis=0)
    c = pd.melt(c, id_vars="Elapsed", value_vars=c.columns[1:], var_name="Lines", value_name="Fold-change confluency")
    c["Condition"] = [s.split(" ")[1].split(" ")[0] for s in c["Lines"]]
    c["Inh_concentration"] = [s[4:].split(" ")[1] for s in c["Lines"]]
    c["Lines"] = [s.split(" ")[0] for s in c["Lines"]]
    c = c[["Elapsed", "Lines", "Condition", "Inh_concentration", "Fold-change confluency"]]
    c = c[c["Elapsed"] >= itp]
    c["IC_n"] = [float(s.split(units)[0]) for s in c["Inh_concentration"]]
    return c.sort_values(by="IC_n").drop("IC_n", axis=1)


def MeanTRs(X):
    """Merge technical replicates of 2 BR by taking the mean."""
    idx = [np.arange(0, 6) + i for i in range(1, X.shape[1], 12)]
    for i in idx:
        for j in i:
            X.iloc[:, j] = X.iloc[:, [j, j + 6]].mean(axis=1)
            X.drop(X.columns[j + 6], axis="columns")

    return X.drop(X.columns[[j + 6 for i in idx for j in i]], axis="columns")


def plotHyerGeomTestDasDRGenes(ax):
    """Data from https://systems.crump.ucla.edu/hypergeometric/index.php where:
    - N = common peptides across both expts
    - M = cluster 4 among N
    - s = das responding among N
    - k = overlap
    Counts generated using GenerateHyperGeomTestParameters()."""
    hg = pd.DataFrame()
    hg["Cluster"] = np.arange(5) + 1
    hg["p_value"] = [0.515, 0.179, 0.244, 0.0013, 0.139]
    sns.barplot(data=hg, x="Cluster", y="p_value", ax=ax, color="darkblue", **{"linewidth": 1}, **{"edgecolor": "black"})
    ax.set_title("Enrichment of Das-responsive Peptides")
    ax.set_ylim((0, 0.55))
    for index, row in hg.iterrows():
        ax.text(row.Cluster - 1, row.p_value + 0.01, round(row.p_value, 3), color='black', ha="center")


def GenerateHyperGeomTestParameters(A, X, dasG, cluster):
    """Generate parameters to calculate p-value for under- or over-enrichment based on CDF of the hypergeometric distribution."""
    N = list(set(A["Gene"]).intersection(set(X["Gene"])))
    cl = axl_ms[A["Cluster"] == cluster]
    M = list(set(cl["Gene"]).intersection(set(N)))
    s = list(set(dasG).intersection(set(N)))
    k = list(set(s).intersection(set(M)))
    return (len(k), len(s), len(M), len(N))


def plot_DasDR_timepoint(ax, inhibitor, units, time=96):
    """Plot dasatinib DR at specified time point."""
    if inhibitor == "Dasatinib":
        br1 = pd.read_csv("msresist/data/Validations/CellGrowth/Dasatinib_Dose_BR3.csv")
        br1.columns = [col.split(".1")[0].strip() for col in br1.columns]
        inh = [br1.groupby(lambda x:x, axis=1).mean(), pd.read_csv("msresist/data/Validations/Experimental/DoseResponses/Dasatinib_2fixed.csv")]
        units = "nM"
    elif inhibitor == "CX-4945":
        inh = pd.read_csv("CX_4945_BR1 _dose.csv")
        inh.columns = [col.split(".1")[0].strip() for col in inh.columns]
        inh = [inh.groupby(lambda x:x, axis=1).mean()]
        units = "uM"
    elif inhibitor == "Volasertib":
        inh = pd.read_csv("msresist/data/Validations/CellGrowth/Volasertib_Dose_BR1.csv")
        inh.columns = [col.split(".1")[0].strip() for col in inh.columns]
        inh = [inh.groupby(lambda x:x, axis=1).mean()]
        units = "nM"
    data = transform_YAPviability_data(inh, inhibitor, units)
    tp = data[data["Elapsed"] == time]
    sns.lineplot(data=tp, x="Inh_concentration", y="Fold-change confluency", hue="Lines", style="Condition", ci=68, ax=ax)
    ax.set_xlabel("[" + inhibitor + "]")


def plot_pAblSrcYap(ax):
    """Plot luminex p-signal of p-ABL, p-SRC, and p-YAP 127."""
    mfi_AS = pd.read_csv("msresist/data/Validations/Luminex/DasatinibDR_newMEK_lysisbuffer.csv")
    mfi_AS = pd.melt(mfi_AS, id_vars=["Treatment", "Line", "Lysis_Buffer"], value_vars=["p-MEK", "p-YAP", "p-ABL", "p-SRC"], var_name="Protein", value_name="p-Signal")
    mfi_YAP = pd.read_csv("msresist/data/Validations/Luminex/DasatinibDR_pYAP127_check.csv")
    mfi_YAP = pd.melt(mfi_YAP, id_vars=["Treatment", "Line", "Lysis_Buffer"], value_vars=["p-MEK", "p-YAP(S127)"], var_name="Protein", value_name="p-Signal")
    abl = mfi_AS[(mfi_AS["Protein"] == "p-ABL") & (mfi_AS["Lysis_Buffer"] == "RIPA")].iloc[:-1, :]
    abl["Treatment"] = [t.replace("A", "(A)") for t in abl["Treatment"]]
    abl["Treatment"][6:] = abl["Treatment"][1:6]
    src = mfi_AS[(mfi_AS["Protein"] == "p-SRC") & (mfi_AS["Lysis_Buffer"] == "NP-40")].iloc[:-2, :]
    src["Treatment"] = [t.replace("A", "(A)") for t in src["Treatment"]]
    src["Treatment"][6:] = src["Treatment"][1:6]
    yap = mfi_YAP[(mfi_YAP["Protein"] == "p-YAP(S127)") & (mfi_YAP["Lysis_Buffer"] == "RIPA")].iloc[:-1, :]
    yap["Treatment"] = [t.replace("A", "(A)") for t in yap["Treatment"]]
    yap["Treatment"][6:] = yap["Treatment"][1:6]

    sns.barplot(data=abl, x="Treatment", y="p-Signal", hue="Line", ax=ax[0])
    ax[0].set_title("p-ABL")
    ax[0].set_xticklabels(abl["Treatment"][:6], rotation=90)
    sns.barplot(data=src, x="Treatment", y="p-Signal", hue="Line", ax=ax[1])
    ax[1].set_title("p-SRC")
    ax[1].set_xticklabels(src["Treatment"][:6], rotation=90)
    sns.barplot(data=yap, x="Treatment", y="p-Signal", hue="Line", ax=ax[2])
    ax[2].set_title("p-YAP S127")
    ax[2].set_xticklabels(yap["Treatment"][:6], rotation=90)
