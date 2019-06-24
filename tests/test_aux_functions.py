from fhhps.estimator import *


def test_get_coefficient_cond_means():
    """
    Checks if function get_coefficient_cond_means is able to retrieve the
    correct conditional moments when we pass it the *oracle* shock and output
    conditional moments. Note nothing is being estimated in this test.
    """
    fake = generate_data(10000)
    X = fake["df"][["X1", "X2", "X3"]].values
    Z = fake["df"][["Z1", "Z2", "Z3"]].values
    output_cond_means = get_true_output_cond_means(fake)
    shock_means = get_true_shock_means(fake)

    estimate = get_coef_cond_means(X, Z, output_cond_means, shock_means)
    truth = get_true_coef_cond_means(fake)
    error = np.abs(estimate - truth)
    assert np.max(error) < 1e-6


def test_get_coefficient_cond_cov():
    """
    Checks the internal consistency between get_coefficient_cond_cov and
    the get_true_* functions for second moments
    """

    n = 100
    fake = generate_data(n, seed=123)
    X = fake["df"][["X1", "X2", "X3"]].values
    Z = fake["df"][["Z1", "Z2", "Z3"]].values

    shock_cov = get_true_shock_cov(fake)
    output_cond_cov = get_true_output_cond_cov(fake)
    truth = get_true_coef_cond_cov(fake)

    # computed passing oracle values
    computed = get_coef_cond_cov(X, Z, output_cond_cov, shock_cov)
    error = computed - truth

    assert np.all(np.abs(error) < 1e-10)


def test_get_true_coef_cov_with_oracle():
    """
    Checks that if everything is perfectly estimated, then variances and covariances
    of random coefficients are computed perfectly as well.
    """
    n = 1000000
    fake = generate_data(n, seed=123)
    X = fake["df"][["X1", "X2", "X3"]].values
    Z = fake["df"][["Z1", "Z2", "Z3"]].values
    shock_cov = get_true_shock_cov(fake)
    output_cond_cov = get_true_output_cond_cov(fake)

    coef_cond_means = get_true_coef_cond_means(fake)
    coef_cond_cov = get_coef_cond_cov(X, Z, output_cond_cov, shock_cov)
    valid = get_valid_cond_cov(X, Z, 1.)
    estimate = get_coef_cov(coef_cond_means, coef_cond_cov, valid)
    truth = get_true_coef_cov(fake)

    # computed passing oracle values
    error = estimate - truth
    assert np.all(np.abs(error) < 1e-2)
