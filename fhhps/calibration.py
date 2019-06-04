from scipy.optimize import minimize_scalar
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error

from fhhps.estimator import *


def calibrate_shock_means_const(n, num_sims=40, shock_alpha=0.2):
    const = np.empty((num_sims, 2))

    for s in range(num_sims):

        if s % 10 == 0:
            logging.info("calibrating shock means: {s}".format(s=s))

        fake = generate_data(n)
        data = fake["df"]
        truth1 = fake["means"][["U2", "V2", "W2"]]
        truth2 = fake["means"][["U3", "V3", "W3"]]
        X = data[["X1", "X2", "X3"]].values
        Z = data[["Z1", "Z2", "Z3"]].values
        Y = data[["Y1", "Y2", "Y3"]].values

        def obj1(coef):
            means = get_shock_means(X, Z, Y, t=1, bw=coef * n ** -shock_alpha)
            return mean_squared_error(means, truth1)

        def obj2(coef):
            means = get_shock_means(X, Z, Y, t=2, bw=coef * n ** -shock_alpha)
            return mean_squared_error(means, truth2)

        const[s, 0] = minimize_scalar(fun=obj1, bounds=[1e-6, 20.], method="Bounded").x
        const[s, 1] = minimize_scalar(fun=obj2, bounds=[1e-6, 20.], method="Bounded").x

    return const


def calibrate_shock_variance_const(n, num_sims=40, shock_means_const=5., shock_alpha=0.2):
    const = np.empty((num_sims, 2))

    for s in range(num_sims):

        if s % 10 == 0:
            logging.info("calibrating shock variances: {s}".format(s=s))

        fake = generate_data(n)
        data = fake["df"]
        truth1 = np.hstack([fake["variances"][["U2", "V2", "W2"]],
                            fake["cov"].loc["U2", "V2"],
                            fake["cov"].loc["U2", "W2"],
                            fake["cov"].loc["V2", "W2"]])
        truth2 = np.hstack([fake["variances"][["U3", "V3", "W3"]],
                            fake["cov"].loc["U3", "V3"],
                            fake["cov"].loc["U3", "W3"],
                            fake["cov"].loc["V3", "W3"]])
        X = data[["X1", "X2", "X3"]].values
        Z = data[["Z1", "Z2", "Z3"]].values
        Y = data[["Y1", "Y2", "Y3"]].values

        bw1 = shock_means_const * n ** -shock_alpha
        m1 = get_shock_means(X, Z, Y, t=1, bw=bw1)
        m2 = get_shock_means(X, Z, Y, t=2, bw=bw1)

        def obj1(coef):
            bw = coef * n ** -shock_alpha
            s1 = get_shock_second_moments(X, Z, Y, 1, bw=bw)
            v1 = center_shock_second_moments(m1, s1)
            return mean_squared_error(v1, truth1)

        def obj2(coef):
            bw = coef * n ** -shock_alpha
            s2 = get_shock_second_moments(X, Z, Y, 2, bw=bw)
            v2 = center_shock_second_moments(m2, s2)
            return mean_squared_error(v2, truth2)

        const[s, 0] = minimize_scalar(fun=obj1, bounds=[1e-6, 20.], method="Bounded").x
        const[s, 1] = minimize_scalar(fun=obj2, bounds=[1e-6, 20.], method="Bounded").x

    return const


def calibrate_coeff_mean_params(n, num_sims=40,
                                shock_const=5.,
                                shock_alpha=0.2,
                                coef_alpha=0.5):
    def obj(params):
        coef_const, censor1_const = params
        try:
            fake = generate_data(n)
            data = fake["df"]
            truth1 = fake["means"][["A1", "B1", "C1"]]

            est = FHHPSEstimator(shock_const=shock_const,
                                 shock_alpha=shock_alpha,
                                 coef_const=coef_const,
                                 censor1_const=censor1_const,
                                 coef_alpha=coef_alpha)
            est.add_data(X=data[["X1", "X2", "X3"]],
                         Y=data[["Y1", "Y2", "Y3"]],
                         Z=data[["Z1", "Z2", "Z3"]])
            est.fit_shock_means()
            est.fit_output_cond_means()
            est.fit_coefficient_means()
            mse = mean_squared_error(est.coefficient_means, truth1)
            print(coef_const, censor1_const, mse)
            return mse
        except ValueError:
            print("Too low!")
            return np.nan

    train_random_params = np.column_stack([
        np.random.uniform(10, 50, size=num_sims),
        np.random.uniform(0.01, 3, size=num_sims)])
    mse = np.empty(shape=num_sims)
    for i, p in enumerate(train_random_params):
        mse[i] = obj(p)
    valid = np.isfinite(mse)

    # Predict the argmin parameter
    reg = KernelRidge(kernel="poly", degree=4).fit(train_random_params[valid], mse[valid])
    test_random_params = np.column_stack([
        np.random.uniform(1, 50, size=10000),
        np.random.uniform(0.01, 5, size=10000)])
    best_idx = reg.predict(test_random_params).argmin()
    coef_const, censor1_const = test_random_params[best_idx]
    results = dict(train_random_params=train_random_params,
                   mse=mse,
                   valid=valid,
                   coef_const=coef_const,
                   censor1_const=censor1_const)
    return results


if __name__ == "__main__":
    # shock_const2000 = calibrate_shock_means_const(n=2000)
    # shock_const5000 = calibrate_shock_means_const(n=5000)
    import matplotlib.pyplot as plt

    res = calibrate_coeff_mean_params(2000, num_sims=30)

    fig, axs = plt.subplots(1, 2, figsize=(15, 4))
    axs[0].scatter(res["train_random_params"][:, 0], res['mse'])
    axs[1].scatter(res["train_random_params"][:, 1], res['mse'])
    plt.show()
