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

    estimate = get_coefficient_cond_means(X, Z, output_cond_means, shock_means)
    truth = get_true_coef_cond_means(fake)
    error = np.abs(estimate - truth)
    assert np.max(error) < 1e-6
