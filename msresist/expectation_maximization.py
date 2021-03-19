"""Main Expectation-Maximization functions using gmm and binomial or pam250 to determine cluster assginments.
EM Co-Clustering Method using a PAM250 or a Binomial Probability Matrix """

import numpy as np
from sklearn.mixture import GaussianMixture
from statsmodels.multivariate.pca import PCA
from pomegranate import GeneralMixtureModel, NormalDistribution, IndependentComponentsDistribution


def EM_clustering_repeat(nRepeats=3, *params):
    output = EM_clustering(*params)

    for _ in range(nRepeats):
        output_temp = EM_clustering(*params)

        # Use the new result if it's better
        if output_temp[0] > output[0]:
            output = output_temp

    return output


def EM_clustering(data, info, ncl: int, seqWeight: float, seqDist=None, gmmIn=None):
    """ Compute EM algorithm to cluster MS data using both data info and seq info.  """
    d = np.array(data.T)

    # Indices for looking up probabilities later.
    idxx = np.atleast_2d(np.arange(d.shape[0]))

    # In case we have missing data, use SVD-EM to fill it for initialization
    pc = PCA(d, ncomp=4, missing="fill-em", method="nipals", tol=1e-9, standardize=False, demean=False, normalize=False)

    # Add a dummy variable for the sequence information
    d = np.hstack((d, idxx.T))

    # Setup weights for distributions
    seqWarr = np.ones(d.shape[1])
    seqWarr[-1] = seqWeight
    seqWarr /= np.sum(seqWarr)

    for _ in range(10):
        # Solve with imputation first
        km = GaussianMixture(ncl, tol=1e-9, covariance_type="diag")
        km.fit(pc._adjusted_data)
        gpp = km.predict_proba(pc._adjusted_data)

        if gmmIn is None:
            # Initialize model
            dists = list()
            for ii in range(ncl):
                nDist = [NormalDistribution(km.means_[ii, jj], km.covariances_[ii, jj], min_std=0.01) for jj in range(d.shape[1] - 1)]

                if isinstance(seqDist, list):
                    nDist.append(seqDist[ii])
                else:
                    nDist.append(seqDist.copy())

                # Setup sequence distribution
                nDist[-1].weightsIn[:] = gpp[:, ii]
                nDist[-1].from_summaries()

                dists.append(IndependentComponentsDistribution(nDist, weights=seqWarr))

            gmm = GeneralMixtureModel(dists)
        else:
            gmm = gmmIn

        gmm.fit(d, max_iterations=2000, verbose=False, stop_threshold=1e-4)
        scores = gmm.predict_proba(d)

        if np.all(np.isfinite(scores)):
            break

    seq_scores = np.exp([dd[-1].logWeights for dd in gmm.distributions])
    avgScore = np.sum(gmm.log_probability(d))

    assert np.all(np.isfinite(scores))
    assert np.all(np.isfinite(seq_scores))

    return avgScore, scores, seq_scores, gmm
