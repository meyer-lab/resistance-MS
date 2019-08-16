""" This scripts handles all the pre-processing required to merge and transform the raw mass spec biological replicates. """

import numpy as np
import pandas as pd
from scipy import stats
from msresist.sequence_analysis import GeneratingKinaseMotifs


###-------------------------- Pre-processing Raw Data --------------------------###

def preprocessing(A_r, B_r, C_r, motifs=False, Vfilter=False, FCfilter=False, log2T=False):
    """ Input: Raw MS bio-replicates. Output: Mean-centered merged data set.
    1. Concatenation, 2. log-2 transformation, 3. Mean-Center, 4. Merging, 5. Fold-change,
    6. Filters: 'Vfilter' filters by correlation when 2 overlapping peptides or std cutoff if >= 3.
    Note 1: 'motifs' redefines peptide sequences as XXXXXyXXXXX which affects merging.
    Note 2: Data is converted back to linear scale before filtering so 'log2T=True' to use log-scale for analysis."""
    ABC = pd.concat([A_r, B_r, C_r])
    ABC_log = Log2T(ABC.copy())
    ABC_conc_mc = MeanCenter(ABC_log, logT=False)

    ABC_names = FormatName(ABC_conc_mc)
    ABC_conc_mc['Master Protein Descriptions'] = ABC_names

    if motifs:
        ABC_seqs = FormatSeq(ABC_conc_mc)
        ABC_conc_mc['peptide-phosphosite'] = ABC_seqs

        directory = "./msresist/data/Sequence_analysis/"
        names, motifs = GeneratingKinaseMotifs(directory + "FaFile.fa", ABC_names, ABC_seqs, directory + "MatchedFaFile.fa", directory + "proteome_uniprot.fa")
        ABC_conc_mc['peptide-phosphosite'] = motifs
        ABC_conc_mc['Master Protein Descriptions'] = names

    ABC_merged = MergeDfbyMean(ABC_conc_mc.copy(), A_r.columns[2:], ['Master Protein Descriptions', 'peptide-phosphosite'])
    ABC_merged = ABC_merged.reset_index()[A_r.columns]
    ABC_merged = LinearScale(ABC_merged)
    ABC_mc = FoldChangeToControl(ABC_merged)

    if Vfilter:
        ABC_conc_mc = LinearScale(ABC_conc_mc)
        ABC_conc_mc_fc = FoldChangeToControl(ABC_conc_mc)
        NonRecPeptides, CorrCoefPeptides, StdPeptides = MapOverlappingPeptides(ABC_conc_mc_fc)

        NonRecTable = BuildMatrix(NonRecPeptides, ABC_conc_mc_fc)

        CorrCoefPeptides = BuildMatrix(CorrCoefPeptides, ABC_conc_mc_fc)
        DupsTable = CorrCoefFilter(CorrCoefPeptides)
        DupsTable = MergeDfbyMean(CorrCoefPeptides, DupsTable.columns[2:], ['Master Protein Descriptions', 'peptide-phosphosite'])
        DupsTable = DupsTable.reset_index()[A_r.columns]

        StdPeptides = BuildMatrix(StdPeptides, ABC_conc_mc_fc)
        TripsTable = TripsMeanAndStd(StdPeptides, A_r.columns)
        TripsTable = FilterByStdev(TripsTable)
        TripsTable.columns = A_r.columns

#         ABC_mc = pd.concat([DupsTable, TripsTable])   # Leaving non-overlapping peptides out
        ABC_mc = pd.concat([NonRecTable, DupsTable, TripsTable])    # Including non-overlapping peptides

    if FCfilter:
        ABC_mc = FoldChangeFilter(ABC_mc)

    if log2T:
        ABC_mc = Log2T(ABC_mc)

    return ABC_mc


def MergeDfbyMean(X, values, index):
    """ Compute mean across duplicates. """
    return pd.pivot_table(X, values=values, index=index, aggfunc=np.mean)


def LinearScale(X):
    """ Convert to linear from log2-scale. """
    X.iloc[:, 2:] = np.power(2, X.iloc[:, 2:])
    return X


def Log2T(X):
    """ Convert to log2 scale keeping original sign. """
    X.iloc[:, 2:] = np.sign(X.iloc[:, 2:]).multiply(np.log2(abs(X.iloc[:, 2:])), axis=0)
    return X


def FoldChangeToControl(X):
    """ Convert to fold-change to control. """
    X.iloc[:, 2:] = X.iloc[:, 2:].div(X.iloc[:, 2], axis=0)
    return X


def MeanCenter(X, logT=False):
    """ Mean centers each row of values. logT also optionally log2-transforms. """
    if logT:
        X.iloc[:, 2:] = np.log2(X.iloc[:, 2:].values)

    X.iloc[:, 2:] = X.iloc[:, 2:].sub(X.iloc[:, 2:].mean(axis=1), axis=0)
    return X


def VarianceFilter(X, varCut=0.1):
    """ Filter rows for those containing more than cutoff variance. Variance across conditions per peptide.
    Note this should only be used with log-scaled, mean-centered data. """
    Xidx = np.var(X.iloc[:, 2:].values, axis=1) > varCut
    return X.iloc[Xidx, :]  # .iloc keeps only those peptide labeled as "True"


def FoldChangeFilter(X):
    """ Filter rows for those containing more than a two-fold change.
    Note this should only be used with linear-scale data normalized to the control. """
    Xidx = np.any(X.iloc[:, 2:].values <= 0.5, axis=1) | np.any(X.iloc[:, 2:].values >= 2.0, axis=1)
    return X.iloc[Xidx, :]


def FormatName(X):
    """ Keep only the general protein name, without any other accession information """
    names = []
    list(map(lambda v: names.append(v.split("OS")[0]), X.iloc[:, 1]))
    return names


def FormatSeq(X):
    """ Deleting -1/-2 for mapping to uniprot's proteome"""
    seqs = []
    list(map(lambda v: seqs.append(v.split("-")[0]), X.iloc[:, 0]))
    return seqs


###------------ Filter by variance (stdev or range/pearson's) ------------------###


def MapOverlappingPeptides(ABC):
    """ Find recurrent peptides across biological replicates. Grouping those showing up 2 to later calculate
    correlation, those showing up >= 3 to take the std. Those showing up 1 can be included or not in the final data set.
    Final dfs are formed by 'Name', 'Peptide', '#Recurrences'. """
    dups = pd.pivot_table(ABC, index=['Master Protein Descriptions', 'peptide-phosphosite'], aggfunc="size").sort_values()
#     dups_counter = {i: list(dups).count(i) for i in list(dups)}
    dups = pd.DataFrame(dups).reset_index()
    dups.columns = [ABC.columns[1], ABC.columns[0], "Recs"]
    NonRecPeptides = dups[dups["Recs"] == 1]
    RangePeptides = dups[dups["Recs"] == 2]
    StdPeptides = dups[dups["Recs"] >= 3]
    return NonRecPeptides, RangePeptides, StdPeptides


def BuildMatrix(peptides, ABC):
    """ Map identified recurrent peptides in the concatenated data set to generate complete matrices with values.
    If recurrent peptides = 2, the correlation coefficient is included in a new column. """
    peptideslist = []
    corrcoefs = []
    for idx, seq in enumerate(peptides.iloc[:, 1]):
        name = peptides.iloc[idx, 0]
        pepts = ABC.reset_index().set_index(["peptide-phosphosite", "Master Protein Descriptions"], drop=False).loc[seq, name]
        pepts = pepts.iloc[:, 1:]
        names = pepts.iloc[:, 1]
        if name == "(blank)":
            continue
        elif len(pepts) == 1:
            peptideslist.append(pepts.iloc[0, :])
        elif len(pepts) == 2 and len(set(names)) == 1:
            corrcoef, _ = stats.pearsonr(pepts.iloc[0, 2:], pepts.iloc[1, 2:])
            for i in range(len(pepts)):
                corrcoefs.append(corrcoef)
                peptideslist.append(pepts.iloc[i, :])
        elif len(pepts) >= 3 and len(set(names)) == 1:
            for i in range(len(pepts)):
                peptideslist.append(pepts.iloc[i, :])
        else:
            print("check this", pepts)

    if corrcoefs:
        matrix = pd.DataFrame(peptideslist).reset_index(drop=True)
        matrix = matrix.assign(CorrCoefs=corrcoefs)

    else:
        matrix = pd.DataFrame(peptideslist).reset_index(drop=True)

    return matrix


def CorrCoefFilter(X, corrCut=0.5):
    """ Filter rows for those containing more than a correlation threshold. """
    Xidx = X.iloc[:, 12].values >= corrCut
    return X.iloc[Xidx, :]


def DupsMeanAndRange(duplicates, header):
    """ Merge all duplicates by mean and range across conditions. Note this builds a multilevel header
    meaning we have 2 values for each condition (eg within Erlotinib -> Mean | Range). """
    func_dup = {}
    for i in header[2:]:
        func_dup[i] = np.mean, np.ptp
    ABC_dups_avg = pd.pivot_table(duplicates, values=header[2:], index=['Master Protein Descriptions', 'peptide-phosphosite'], aggfunc=func_dup)
    ABC_dups_avg = ABC_dups_avg.reset_index()[header]
    return ABC_dups_avg


def TripsMeanAndStd(triplicates, header):
    """ Merge all triplicates by mean and standard deviation across conditions. Note this builds a multilevel header
    meaning we have 2 values for each condition (eg within Erlotinib -> Mean | Std). """
    func_tri = {}
    for i in header[2:]:
        func_tri[i] = np.mean, np.std
    ABC_trips_avg = pd.pivot_table(triplicates, values=header[2:], index=['Master Protein Descriptions', 'peptide-phosphosite'], aggfunc=func_tri)
    ABC_trips_avg = ABC_trips_avg.reset_index()[header]
    return ABC_trips_avg


def FilterByRange(X, rangeCut=0.4):
    """ Filter rows for those containing more than a range threshold. """
    Rg = X.iloc[:, X.columns.get_level_values(1) == 'ptp']
    Xidx = np.all(Rg.values <= rangeCut, axis=1)
    return X.iloc[Xidx, :]


def FilterByStdev(X, stdCut=0.5):
    """ Filter rows for those containing more than a standard deviation threshold. """
    Stds = X.iloc[:, X.columns.get_level_values(1) == 'std']
    Xidx = np.all(Stds.values <= stdCut, axis=1)
    Means = pd.concat([X.iloc[:, 0], X.iloc[:, 1], X.iloc[:, X.columns.get_level_values(1) == "mean"]], axis=1)
    return Means.iloc[Xidx, :]
