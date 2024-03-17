import numpy as np
from tqdm import tqdm

# TODO create subclass to be able to have all optimization results under the same instance
# TODO add follow-up NR/GD iteration (require derivative function)


class Optimizer:
    def __init__(self, fun, search_range, n_particles, n_iterations, n_best=1, vel_ini_scale=0.5, is_adaptive=True,
                 debug_mode=False, params=None):
        #
        # basic parameters
        self.fun = fun  # optimized function
        self.range = np.array(search_range)  # search range
        self.n_pt = n_particles  # number of particles
        self.n_it = n_iterations  # number of iterations
        self.n_best = n_best  # number of best (lowest) values to return
        self.v_ini = vel_ini_scale  # scaling factor of initial velocity
        self.n_dim = self.range.shape[0]  # input dimension of optimized function
        if len(self.range.shape) == 1:
            dim = self.range[1] - self.range[0]
            low = self.range[0]
            up = self.range[1]
        else:
            assert self.range.shape[0] == self.n_dim
            assert self.range.shape[1] == 2
            dim = self.range[:, 1] - self.range[:, 0]
            low = self.range[:, 0]
            up = self.range[:, 1]
        self.dim = dim.reshape((self.n_dim, 1))  # dimensions of search range
        self.low = low.reshape((self.n_dim, 1))  # lowermost corner of search range
        self.up = up.reshape((self.n_dim, 1))  # uppermost corner of search range
        self.t = np.arange(0, self.n_it + 1)  # time grid
        self.results = {
            'g_bests': np.zeros((self.n_dim, self.n_it)),
            'fg_bests': np.zeros_like(self.t)}  # all results are stored here
        self.debug = True if debug_mode else False
        #
        # setting up PSO parameters
        if is_adaptive:
            self.pso_pars = None
            if params is None:
                self.apso_pars = {
                    'Vc': [
                        [0, round(self.n_it / 5), round(4 * self.n_it / 5), self.n_it],
                        [5, 5, 25, 25]
                    ],
                    'rho1': [
                        [0, round(self.n_it / 5),  round(self.n_it / 2), round(4 * self.n_it / 5), self.n_it],
                        [0.1, 0.1, 0.8, 0.1, 0.1]
                    ],
                    'F': [
                        [0, round(self.n_it / 5), round(self.n_it / 5) + 1, round(4 * self.n_it / 5),
                         round(4 * self.n_it / 5) + 1, self.n_it],
                        [0.25, 0.25, 1, 1, 25, 25]
                    ]
                }
            else:
                # TODO sanity check of params
                self.apso_pars = params
        else:
            self.apso_pars = None
            if params is None:
                self.pso_pars = {
                    'w': [
                        [0, self.n_it],
                        [0.712, 0.712]
                    ],
                    'c1': [
                        [0, self.n_it],
                        [1.712, 1.712]
                    ],
                    'c2': [
                        [0, self.n_it],
                        [1.712, 1.712]
                    ]
                }
            else:
                # TODO sanity check of params
                self.pso_pars = params

        if self.apso_pars is not None:
            self.pars = self.transform_pars(*self.compute_all_pars(self.t, self.apso_pars))
        elif self.pso_pars is not None:
            self.pars = self.compute_all_pars(self.t, self.pso_pars)

    def compute_all_pars(self, t, pars):
        # computing parameter values over all iteration steps
        out = ()
        for key in pars:
            pars[key] = np.array(pars[key])
            out += (np.interp(t, pars[key][0, :], pars[key][1, :]),)
        return out

    def transform_pars(self, Vc, rho1, F):
        # transforming apso parameters back to pso parameter space
        alp = np.sqrt(F)
        m1 = np.power(alp + 1, 2) * (1 + 3 * alp + np.power(alp, 2))
        m2 = np.power(alp + 1, 2) * (2 + 3 * alp + 2 * np.power(alp, 2))
        w = (m1 + m2 * rho1 - (1 - rho1) / Vc) / (m2 + m1 * rho1 + (1 - rho1) / Vc)
        c = 2 * (1 - rho1) * (w + 1) / (alp + 1)
        return w, c, c * alp

    def initialize(self):
        """
        Creates the initial swarm of particles and initializes the Canonical Particle Swarm Optimization (CPSO)
        iteration.
        :return:    initial (pos, vel, p_best, g_best, fp_best) values for CPSO
        """
        # generating particle positions and velocities
        pos = self.dim * np.random.uniform(0, 1, (self.n_dim, self.n_pt)) + self.low
        vel = 2 * self.dim * np.random.uniform(0, 1, (self.n_dim, self.n_pt)) - self.dim
        p_best, fp_best = pos, self.fun(pos)
        # initializing global best position
        g_best = p_best[:, np.argmin(fp_best)]
        return pos, self.v_ini * vel, p_best, g_best, fp_best

    def step(self, pos, vel, p_best, g_best, fp_best, t):
        """
            Computes a synchronous single Canonical Particle Swarm Optimization (CPSO) step.
            :param pos:     2D numpy array: (d, n), position vector for each particle
            :param vel:     2D numpy array: (d, n), velocity vector for each particle
            :param p_best:  2D numpy array: (d, n), personal best position of each particle
            :param g_best:  1D numpy array: (d), global best position of the entire swarm
            :param fp_best: 1D numpy array: (n), objective function value at personal best position
            :param t:       int: iteration time step
            :return: new (pos, vel, p_best, g_best, fp) values for CPSO
            """
        w, c1, c2 = self.pars
        d, n = pos.shape
        #
        # updating position and velocity of particles

        vel = w[t] * vel + c1[t] * np.random.uniform(0, 1, (d, n)) * (p_best - pos) + \
              c2[t] * np.random.uniform(0, 1, (d, n)) * (g_best.reshape((d, -1)) - pos)
        pos = pos + vel
        # reassigning values that violate constraints
        ind_low = pos < self.low
        ind_up = pos > self.up
        pos[ind_low] = np.broadcast_to(self.low, (d, n))[ind_low]
        pos[ind_up] = np.broadcast_to(self.up, (d, n))[ind_up]
        #
        # updating personal and global best positions

        # evaluate objective function at all particle positions
        f_vals = self.fun(pos)
        # indices of particles whose personal best positions are to be updated
        ind_update = f_vals < fp_best
        # update personal best positions
        p_best[:, ind_update] = pos[:, ind_update]
        # update objective function values corresponding to personal best positions
        fp_best[ind_update] = f_vals[ind_update]
        # update global best position
        ind_best = np.argmin(fp_best)
        g_best = p_best[:, ind_best]
        fg_best = fp_best[ind_best]
        return pos, vel, p_best, g_best, fp_best, fg_best

    def run(self):
        """
        Carries out the PSO iteration.
        """
        pos, vel, p_best, g_best, fp = self.initialize()
        for t in tqdm(range(self.n_it)):
            pos, vel, p_best, g_best, fp, fg = self.step(pos, vel, p_best, g_best, fp, t)
            self.results['g_bests'][:, t] = g_best
            self.results['fg_bests'][t] = fg
        ind_best = np.argsort(fp)[:self.n_best]
        self.results['p_best_final'] = p_best[:, ind_best]
        self.results['fp_best_final'] = fp[ind_best]
        return self.results['p_best_final'], self.results['fp_best_final']

    def get_pbests(self):
        if self.debug:
            return self.p_bests
        else:
            print('No personal bests available, due to debug mode not being enabled.')

    def get_pos(self):
        if self.debug:
            return self.pos
        else:
            print('No positions available, due to debug mode not being enabled.')

    def get_vel(self):
        if self.debug:
            return self.vel
        else:
            print('No velocities available, due to debug mode not being enabled.')

    def get_results(self):
        return self.results

    def get_parameters(self):
        return self.pars

    def plot_parameters(self):
        return None

    def animate(self):
        return None
