"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
# import additional ...
import sklearn.gaussian_process as gp
from scipy.stats import norm
import matplotlib.pyplot as plt

# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        self.k = SAFETY_THRESHOLD
        sigma_f = 0.15
        sigma_v = 0.0001

        self.kernel_f = gp.kernels.Matern(nu=2.5)
        self.kernel_v = gp.kernels.DotProduct() + gp.kernels.Matern(nu=2.5) * gp.kernels.ConstantKernel(4)
        self.gp_f = gp.GaussianProcessRegressor(kernel=self.kernel_f, alpha=sigma_f**2, n_restarts_optimizer=10)
        self.gp_v = gp.GaussianProcessRegressor(kernel=self.kernel_v, alpha=sigma_v**2, n_restarts_optimizer=10)
        self.data_x = np.array([]).reshape(-1, DOMAIN.shape[0])
        self.data_f = np.array([]).reshape(-1, DOMAIN.shape[0])
        self.data_v = np.array([]).reshape(-1, DOMAIN.shape[0])
        

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.
        if self.data_x.shape[0] == 0:
            return get_initial_safe_point()
        else:
            return np.atleast_2d(self.optimize_acquisition_function())

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def expected_improvement(self, x):
        mu, sigma = self.gp_f.predict(x.reshape(-1, DOMAIN.shape[0]), return_std=True)
        all_f = self.gp_f.predict(self.data_x, return_std=False)
        max_f = np.max(all_f)
        improv = mu - max_f
       
        return improv * norm.cdf(improv / sigma) + sigma * norm.pdf(improv / sigma)

    def expected_constrained(self, x):
            mu, sigma = self.gp_v.predict(x.reshape(-1, DOMAIN.shape[0]), return_std=True)
            expected = self.k - mu
            return expected * norm.cdf(expected / sigma) + sigma * norm.pdf(expected / sigma)

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.
        # # Calculate the expected improvement 
        # # EI(x) = E[max(0, f(x) - f(x_best))]
        # # Get x* = argmax_x f(x)
        # x_best = self.get_solution()
        # # Get mu(x) and sigma(x) from the GP
        # mu_f, sigma_f = self.gp_f.predict(x, return_std=True)
        # z = (mu_f - x_best) / sigma_f
        # # Calculate the expected improvement
        # af_value = (mu_f - x_best) * norm.cdf(z) + sigma_f * norm.pdf(z)
        # print(af_value)

        #EI
        # t = np.array(self.data_f.max())
        # c_mean, c_std = self.gp_v.predict(x, return_std=True)
        # prop_constraint = norm.cdf((self.k - c_mean) / c_std)
        
        # f_mean, f_std = self.gp_f.predict(x, return_std=True)
        # z_x = (f_mean - t - 0.1) / f_std

        # ei_x = f_std * (z_x * norm.cdf(z_x) + norm.pdf(z_x))

        # return prop_constraint * ei_x

        # UCB
        mu_f, sigma_f = self.gp_f.predict(x, return_std=True)
        res = mu_f + 0.01 * sigma_f

        c_mean, c_std = self.gp_v.predict(x, return_std=True)
        prop_constraint = norm.cdf((self.k - c_mean) / c_std)
        #print(f"prop_constraint: {prop_constraint}, res: {res}")
        return prop_constraint * res

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.
        self.data_x = np.vstack([self.data_x, x])
        self.data_v = np.vstack([self.data_v, v])
        self.data_f = np.vstack([self.data_f, f])
        self.gp_f.fit(self.data_x, self.data_f)
        self.gp_v.fit(self.data_x, self.data_v)

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        pred = self.data_v < self.k
        return self.data_x[np.argmax(self.data_f[pred])]

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        if plot_recommendation:
            x = self.next_recommendation()
            plt.plot(x, self.acquisition_function(x), 'ro', label='Recommendation')


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---


def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.randn()
        cost_val = v(x) + np.random.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
