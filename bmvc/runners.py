from vi_base import run_vi
from gd_base import run_gd
from FullBMVCModel import FullBMVCModel
from PseudoBMVCModel import PseudoBMVCModel


def gen_vi_bmvc_results(X1, X2, R, K,
                       phi=None,
                       phi_mean_prior=0,
                       phi_scale_prior=1,
                       init='kmeans',
                       random_state=0,
                       iter=500,
                       inference='alternating',
                       step_size=0.1,
                       pseudo=False,
                       phi_power=1,
                       track_convergence=False):
    if pseudo:
        model = PseudoBMVCModel(X1,
                               X2,
                               R,
                               K,
                               phi,
                               phi_mean_prior=phi_mean_prior,
                               phi_scale_prior=phi_scale_prior,
                               init=init,
                               random_state=random_state,
                               inference=inference,
                               phi_power=phi_power,
                               track_convergence=track_convergence)
    else:
        model = FullBMVCModel(X1,
                             X2,
                             R,
                             K,
                             phi,
                             phi_mean_prior=phi_mean_prior,
                             phi_scale_prior=phi_scale_prior,
                             init=init,
                             random_state=random_state,
                             inference=inference,
                             phi_power=phi_power,
                             track_convergence=track_convergence)
    vi_params = run_vi(model, step_size=step_size, num_samples=10, num_iters=iter, random_state=random_state)
    mv_z1, mv_z2 = model.get_z()
    return model, vi_params


def gen_gd_bmvc_results(X1, X2, R, K,
                       phi=None,
                       phi_mean_prior=0,
                       phi_scale_prior=1,
                       init='kmeans',
                       random_state=0,
                       iter=500,
                       inference='alternating',
                       step_size=0.1,
                       pseudo=False,
                       phi_power=1,
                       track_convergence=False):
    if pseudo:
        model = PseudoBMVCModel(X1,
                               X2,
                               R,
                               K,
                               phi,
                               phi_mean_prior=phi_mean_prior,
                               phi_scale_prior=phi_scale_prior,
                               init=init,
                               random_state=random_state,
                               inference=inference,
                               phi_power=phi_power,
                               track_convergence=track_convergence)
    else:
        model = FullBMVCModel(X1,
                             X2,
                             R,
                             K,
                             phi,
                             phi_mean_prior=phi_mean_prior,
                             phi_scale_prior=phi_scale_prior,
                             init=init,
                             random_state=random_state,
                             inference=inference,
                             phi_power=phi_power,
                             track_convergence=track_convergence)

    gd_params = run_gd(model, step_size=step_size, num_iters=iter, random_state=random_state)
    mv_z1, mv_z2 = model.get_z()
    return model, gd_params
