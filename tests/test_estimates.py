from fhhps.estimator import *


def test_fit_shock_means():
    n = 5000000
    fake = generate_data(n, seed=1234)
    data = fake["df"]
    truth = get_true_shock_means(fake)
    X = data[["X1", "X2", "X3"]].values
    Z = data[["Z1", "Z2", "Z3"]].values
    Y = data[["Y1", "Y2", "Y3"]].values

    bw = 5 * n ** (-1 / 3)
    estimate = fit_shock_means(X, Z, Y, bw=bw)
    assert np.all(np.abs(estimate - truth) < 0.05)


def test_fit_shock_second_cov():
    n = 5000000
    fake = generate_data(n, seed=1234)
    data = fake["df"]
    truth = get_true_shock_cov(fake)
    X = data[["X1", "X2", "X3"]].values
    Z = data[["Z1", "Z2", "Z3"]].values
    Y = data[["Y1", "Y2", "Y3"]].values

    bw = 2.5 * n ** (-1 / 3)
    estimate = fit_shock_cov(X, Z, Y, get_true_shock_means(fake), bw=bw)
    assert np.all(np.abs(estimate - truth) < 0.05)


def test_fit_output_cond_means():
    """
    Checks that we are able to estimate first conditional
    moments of the output at a reasonable level: avg error is not
    statistically different from zero at alpha=0.01
    """
    np.random.seed(1234)
    coef_bw = .05
    errors = []
    for i in range(40):
        fake = generate_data(1000)
        X = fake["df"][["X1", "X2", "X3"]].values
        Z = fake["df"][["Z1", "Z2", "Z3"]].values
        Y = fake["df"][["Y1", "Y2", "Y3"]].values

        truth = get_true_output_cond_means(fake)
        estimate = fit_output_cond_means(X, Z, Y, bw=coef_bw)
        error = estimate - truth
        errors.append(error.mean(0))

    stats = pd.DataFrame(errors).agg(["mean", "sem"])
    unbiased = np.abs(stats.loc["mean"]) / stats.loc["sem"] < 2.326
    assert np.all(unbiased)


def test_fit_output_cond_cov_with_oracle_means():
    """
    Checks that if we pass it the **true** conditional means, then
    we are able to estimate second conditional
    moments of the output at a reasonable level: avg error is not
    statistically different from zero at alpha=0.01
    """
    np.random.seed(1234)
    coef_bw = .75
    errors = []
    for i in range(40):
        fake = generate_data(1000)
        X = fake["df"][["X1", "X2", "X3"]].values
        Z = fake["df"][["Z1", "Z2", "Z3"]].values
        Y = fake["df"][["Y1", "Y2", "Y3"]].values

        output_cond_means = get_true_output_cond_means(fake)
        estimate = fit_output_cond_cov(X, Z, Y, output_cond_means, bw=coef_bw)
        truth = get_true_output_cond_cov(fake)
        error = estimate - truth
        errors.append(error.mean(0))

    stats = pd.DataFrame(errors).agg(["mean", "sem"])
    unbiased = np.abs(stats.loc["mean"]) / stats.loc["sem"] < 2.326
    assert np.all(unbiased)


def test_fit_output_cond_cov_with_estimated_means():
    """
    Checks that if we pass it the **estimated** conditional means, then
    we are able to estimate second conditional
    moments of the output at a reasonable level: avg error is not
    statistically different from zero at alpha=0.01
    """
    np.random.seed(1234)
    errors = []
    for i in range(40):
        fake = generate_data(1000)
        X = fake["df"][["X1", "X2", "X3"]].values
        Z = fake["df"][["Z1", "Z2", "Z3"]].values
        Y = fake["df"][["Y1", "Y2", "Y3"]].values

        output_cond_means = fit_output_cond_means(X, Z, Y, bw=.75)
        estimate = fit_output_cond_cov(X, Z, Y, output_cond_means, bw=.5)
        truth = get_true_output_cond_cov(fake)
        error = estimate - truth
        errors.append(error.mean(0))
        print(error.mean(0))

    stats = pd.DataFrame(errors).agg(["mean", "sem"])
    unbiased = np.abs(stats.loc["mean"]) / stats.loc["sem"] < 2.326
    assert np.all(unbiased)


# TODO: calibrate bandwidths
def test_fit_coefficient_cond_means():
    n = 1000
    np.random.seed(123)
    shock_bw = 1 * n ** (-1 / 4)
    output_bw = 1 * n ** (-1 / 2)
    censor_const = 1.

    errors = []
    for i in range(20):
        fake = generate_data(n)
        X = fake["df"][["X1", "X2", "X3"]].values
        Z = fake["df"][["Z1", "Z2", "Z3"]].values
        Y = fake["df"][["Y1", "Y2", "Y3"]].values

        # fit shocks
        shock_means = fit_shock_means(X, Z, Y, shock_bw)

        # fit output
        output_cond_means = fit_output_cond_means(X, Z, Y, bw=output_bw)

        # inversion step
        coef_cond_means = get_coef_cond_means(X, Z, output_cond_means, shock_means)

        # comparison
        truth = get_true_coef_cond_means(fake)
        valid = get_valid_cond_means(X, Z, censor_const)

        coef_error = coef_cond_means[valid] - truth[valid]
        errors.append(coef_error.mean(0))

    stats = pd.DataFrame(errors).agg(["mean", "sem"])
    unbiased = np.abs(stats.loc["mean"]) / stats.loc["sem"] < 2.326
    assert np.all(unbiased)
