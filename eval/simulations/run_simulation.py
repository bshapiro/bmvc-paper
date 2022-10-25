import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, grandparentdir)

from simulations import two_cluster_sim, \
                        two_cluster_sim_onto, \
                        two_cluster_sim_missing, \
                        four_cluster_baseline, \
                        generate_data_p_mod_gt, \
                        generate_data_baseline_mod, \
                        compare_clusterings_ari
from pickle import dump
from runners import gen_vi_bmvc_results, gen_gd_bmvc_results
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import scale
import numpy as np

iter = 500
phi_mean_prior = 0
phi_scale_prior = 1
step_size = 0.1
init = 'kmeans'
inference = 'alternating'
pseudo = False

sim = sys.argv[1]  # can be missing, basic, onto or baseline
mod = float(sys.argv[2])  # shift amount
p = float(sys.argv[3])  # randomness parameter

if sim == 'baseline':
    K = 4
else:
    K = 2

for trial in range(10):
    filename = sim + '_s' + str(mod) + '_m' + str(p) + '_t' + str(trial) + '.dump'
    rs = trial

    if sim == 'basic':
        clusters, n1, n2, sigma1, sigma2, R = two_cluster_sim(100, 0.3)
    elif sim == 'onto':
        clusters, n1, n2, sigma1, sigma2, R = two_cluster_sim_onto(100, 0.3)
    elif sim == 'missing':
        clusters, n1, n2, sigma1, sigma2, R = two_cluster_sim_missing(100, 0.3)
    elif sim == 'baseline':
        clusters, n1, n2, sigma1, sigma2, R = four_cluster_baseline(100, 0.3)

    if sim != 'baseline':
        samples1, samples2, labels1, labels2 = generate_data_p_mod_gt(clusters, [n1, n2], [sigma1, sigma2], p, mod, 50, rs=trial)
    else:
        samples1, samples2, labels1, labels2 = generate_data_baseline_mod(clusters, [n1, n2], [sigma1, sigma2], 0.5, mod, 50, rs=trial)
    X1 = scale(samples1)
    X2 = scale(samples2)

    print("Confirming data is scaled...")
    print(np.var(X1, 0))
    print(np.var(X2, 0))
    print(np.mean(X2, 0))
    print(np.mean(X1, 0))

    print("Running MVC...")
    phi = None

    mvc, params = gen_gd_bmvc_results(X1, X2, R, K,
                                     phi=phi,
                                     phi_mean_prior=phi_mean_prior,
                                     phi_scale_prior=phi_scale_prior,
                                     init=init,
                                     random_state=trial,
                                     iter=iter,
                                     inference=inference,
                                     step_size=step_size,
                                     pseudo=pseudo,
                                     phi_power=1)
    mvc_phi = params[-2]

    ari_mvc_view1 = compare_clusterings_ari(labels1, mvc.zA, 'MVC View 1')
    ari_mvc_view2 = compare_clusterings_ari(labels2, mvc.zB, 'MVC View 2')

    if sim != 'baseline':
        print("Running GMM...")
        phi = 0

        gmm, params = gen_gd_bmvc_results(X1, X2, R, K,
                                         phi=phi,
                                         phi_mean_prior=phi_mean_prior,
                                         phi_scale_prior=phi_scale_prior,
                                         init=init,
                                         random_state=trial,
                                         iter=iter,
                                         inference=inference,
                                         step_size=step_size,
                                         pseudo=pseudo,
                                         phi_power=1)

        ari_gmm_view1 = compare_clusterings_ari(labels1, gmm.zA, 'GMM View 1')
        ari_gmm_view2 = compare_clusterings_ari(labels2, gmm.zB, 'GMM View 2')
    else:

        phi = 0

        gmm, params = gen_gd_bmvc_results(X1, X2, R, K,
                                         phi=phi,
                                         phi_mean_prior=phi_mean_prior,
                                         phi_scale_prior=phi_scale_prior,
                                         init=init,
                                         random_state=trial,
                                         iter=iter,
                                         inference=inference,
                                         step_size=step_size,
                                         pseudo=pseudo,
                                         phi_power=1)

        ari_gmm_view1 = compare_clusterings_ari(labels1, gmm.zA, 'GMM View 1')
        ari_gmm_view2 = compare_clusterings_ari(labels2, gmm.zB, 'GMM View 2')

        merged_samples = np.concatenate((np.array(samples1), np.array(samples2)), 1)
        X = scale(merged_samples)
        gmm = GaussianMixture(n_components=K, random_state=rs)
        gmm_z = gmm.fit(X).predict(X)
        merged_ari_gmm_view1 = compare_clusterings_ari(labels1, gmm_z, 'Merged GMM, View 1')
        merged_ari_gmm_view2 = compare_clusterings_ari(labels2, gmm_z, 'Merged GMM, View 2')
        dump((merged_ari_gmm_view1 + merged_ari_gmm_view2) / 2.0, open('sim_results/' + filename.split('.dump')[0] + '_merged.dump', 'wb'))

    total_score = (ari_mvc_view2 + ari_mvc_view1) / 2.0 - (ari_gmm_view2 + ari_gmm_view1) / 2.0
    avg_score = [(ari_mvc_view2 + ari_mvc_view1) / 2.0, (ari_gmm_view2 + ari_gmm_view1) / 2.0]

    dump(((ari_mvc_view2 + ari_mvc_view1) / 2.0, (ari_gmm_view2 + ari_gmm_view1) / 2.0, mvc_phi, mvc.converged, gmm.converged), open('sim_results/' + filename, 'wb'))
