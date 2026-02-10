



def run_spectrum_minimisation(fit_workspace, method="newton", sigma_clip=20):
    """
    """
    my_logger = set_logger(__name__)
    guess = np.asarray(fit_workspace.params.values)

    # fit_workspace.simulation.fast_sim = True
    fit_workspace.chisq(guess)
    if parameters.DISPLAY and (parameters.DEBUG or fit_workspace.live_fit):
        fit_workspace.plot_fit()

    # params_table = np.array([guess])
    my_logger.info(f"\n\tStart guess: {guess}\n\twith {fit_workspace.params.labels}")

    fit_workspace.simulation.fast_sim = False
    fixed = copy.copy(fit_workspace.params.fixed)
    fit_workspace.params.fixed = [True] * len(fit_workspace.params)
    fit_workspace.params.fixed[fit_workspace.params.get_index(r"A1")] = False
    run_minimisation(fit_workspace, method="newton", xtol=1e-3, ftol=100 / fit_workspace.data.size, verbose=False, with_line_search=False)
    fit_workspace.params.fixed = fixed
    run_minimisation_sigma_clipping(fit_workspace, method="newton", xtol=1e-6, ftol=1e-3 / fit_workspace.data.size, sigma_clip=sigma_clip, niter_clip=3, verbose=False)

    # alternate fixing dccd and pixshift with fitting all parameters
    for i in range(3):
        fixed = copy.copy(fit_workspace.params.fixed)
        fit_workspace.params.fixed = [True] * len(fit_workspace.params)
        fit_workspace.params.fixed[6] = False     # reso
        fit_workspace.params.fixed[7] = fixed[7]  # dccd
        fit_workspace.params.fixed[8] = fixed[8]  # pixshift
        run_minimisation_sigma_clipping(fit_workspace, method="newton", xtol=1e-6,
                                    ftol=1e-3 / fit_workspace.data.size, sigma_clip=sigma_clip, niter_clip=3, verbose=False, with_line_search=True)
        fit_workspace.params.fixed = fixed
        run_minimisation_sigma_clipping(fit_workspace, method="newton", xtol=1e-6,
                                    ftol=1e-3 / fit_workspace.data.size, sigma_clip=sigma_clip, niter_clip=3, verbose=False, with_line_search=True)
        
     

    fit_workspace.params.plot_correlation_matrix()
    fit_workspace.plot_fit()
    extra = {"chi2": fit_workspace.costs[-1] / fit_workspace.data.size,
             "date-obs": fit_workspace.spectrum.date_obs,
             "outliers": len(fit_workspace.outliers)}
    fit_workspace.params.extra = extra



def run_minimisation(fit_workspace, xtol=1e-4, ftol=1e-4, niter=50, verbose=False, with_line_search=True, minimizer_method="L-BFGS-B"):
    my_logger = set_logger(__name__)

    bounds = fit_workspace.params.bounds

    nll = lambda params: -fit_workspace.lnlike(params)

    guess = fit_workspace.params.values.astype('float64')
    
    if verbose:
        my_logger.debug(f"\n\tStart guess: {guess}")


    start = time.time()
    run_gradient_descent(fit_workspace, xtol=xtol, ftol=ftol, niter=niter, verbose=verbose, with_line_search=with_line_search)

    if verbose:
        my_logger.debug(f"\n\tNewton: total computation time: {time.time() - start}s")

    fit_workspace.__post_fit__()

    if verbose and parameters.DEBUG:
        fit_workspace.plot_fit()

