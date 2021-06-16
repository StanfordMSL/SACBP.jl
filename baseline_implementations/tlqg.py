import casadi as cs
import numpy as np
from math import pi, e
import time
import warnings


class TLQG_PosRangeLocalization2D:
    def __init__(self, N, dto, Nt, Q, Cs, Cu, u_param_min, u_param_max):
        # parameters
        self.N = N # plan_horizon == N*dt
        self.dto = dto
        self.Nt = Nt # Number of targets to be tracked.
        self.Q = Q
        self.Cs = Cs # only for LQR tracking
        self.Cu = Cu
        self.u_param_min = u_param_min
        self.u_param_max = u_param_max

        # optimization variables and objective
        self.opti = None
        self.U = None
        self.cost = None

        # placeholder symbols
        self._x = cs.MX.sym('x', 2 + 2*self.Nt) # [px_ego, py_ego, px_target_1, py_target_1, ..., px_target_Nt, py_target_Nt]
        # self._x_new = cs.MX.sym('x_new', 2 + 2*self.Nt)
        self._u = cs.MX.sym('u', 2)
        self._v = cs.MX.sym('v', 2*self.Nt) # observation noise vector
        self._p = cs.MX.sym('p', (2*self.Nt)**2) # vectorized covariance matrix
        self._S = cs.MX.sym('S', (2 + 2*self.Nt), (2 + 2*self.Nt)) # LQR riccati matrix

        # function objects
        self.f = TLQG_PosRangeLocalization2D.get_state_trans_func(self._x, self._u, self.dto)
        self.h = TLQG_PosRangeLocalization2D.get_obs_func(self._x, self._v)
        self.obs_cov = TLQG_PosRangeLocalization2D.get_obs_cov_func(self._x)
        self.obs_jacobians = TLQG_PosRangeLocalization2D.get_obs_jacobians_func(self.h, self._x, self._v)
        self.cov_trans = TLQG_PosRangeLocalization2D.get_cov_trans_func(self.f, self.h, self.obs_jacobians, self._x, self._u, self._v, self._p, self.Q, self.obs_cov)
        self.c_stage = TLQG_PosRangeLocalization2D.get_stage_cost_func(self._u, self.Cu, self.dto)
        self.c_term = TLQG_PosRangeLocalization2D.get_term_cost_func(self._x, self._p)
        self.c_term_grad = TLQG_PosRangeLocalization2D.get_c_term_grad_func(self.c_term, self._p)
        self.gradient_control = TLQG_PosRangeLocalization2D.gradient_control(self.cov_trans, self.c_term_grad, self._x, self._p)
        self.lqr_backward = TLQG_PosRangeLocalization2D.get_lqr_backward_func(self.f, self._x, self._u, self._S, self.Cs, self.Cu)

    def setup_nlp(self, μ0, Σ0):
        # initialize variables
        self.opti = cs.Opti()
        self.U = self.opti.variable(2, self.N)
        self.opti.set_initial(self.U, 0)
        p = cs.vec(Σ0)
        p_initial_guess = p
        x = μ0
        x_initial_guess = μ0
        v = np.zeros(μ0.shape[0]-2)
        self.cost = 0.0
        for tt in range(self.N):
            u_initial_guess = self.gradient_control(x_initial_guess, p_initial_guess)
            self.opti.set_initial(self.U[:, tt], u_initial_guess)
            x_new = self.f(x, self.U[:, tt])
            p = self.cov_trans(x, self.U[:, tt], v, p)
            p_initial_guess = self.cov_trans(x_initial_guess, u_initial_guess, v, p_initial_guess)
            x_initial_guess = self.f(x_initial_guess, u_initial_guess)
            x = x_new
            self.cost += self.c_stage(self.U[:, tt])
        self.cost += self.c_term(p)
        # set up minimization problem
        self.opti.minimize(self.cost)
        # control constraints
        self.opti.subject_to(self.opti.bounded(self.u_param_min[0], self.U[0, :], self.u_param_max[0]))
        self.opti.subject_to(self.opti.bounded(self.u_param_min[1], self.U[1, :], self.u_param_max[1]))

    def clear_opti(self):
        self.opti = None
        self.U = None
        self.cost = None

    def solve_tlqg(self, μ0, Σ0, verbose=True, online=True):
        t0 = time.time()
        self.clear_opti()
        self.setup_nlp(μ0, Σ0)
        if not verbose:
            if online: # we care about computation time of the entire T-LQG planning
                solver_options = dict([('print_level', 0), ('max_iter', 50), ('max_cpu_time', 0.2)])
            else:
                solver_options = dict([('print_level', 0), ('max_iter', 3000000), ('max_cpu_time', 3600)])

        else:
            if online:
                solver_options = dict([('max_iter', 50), ('max_cpu_time', 0.2)])
            else:
                solver_options = dict([('max_iter', 3000000), ('max_cpu_time', 3600)])
        self.opti.solver('ipopt', dict(), solver_options)
        x_array = []
        P_array = []
        l_array = []
        L_array = []
        S = self.Cs
        v = np.zeros(μ0.shape[0]-2)
        elapsed = time.time() - t0
        try:
            sol = self.opti.solve()
            t0 = time.time()
            x, P = μ0, Σ0
            x_array.append(x)
            P_array.append(P)
            p = cs.vec(P)
            Nt = (x.shape[0] - 2)//2
            for tt in range(self.N):
                l_array.append(sol.value(self.U[:, tt]))
                x_new = self.f(x, sol.value(self.U[:, tt]))
                p = self.cov_trans(x, sol.value(self.U[:, tt]), v, p)
                x = x_new
                x_array.append(x.full().flatten())
                P_array.append(cs.reshape(p, 2*Nt, 2*Nt).full())
            for tt in reversed(range(self.N)):
                L, S = self.lqr_backward(x_array[tt], l_array[tt], S)
                L_array.insert(0, L.full())
            elapsed += time.time() - t0
        except:
            # solve not finished within time budget or failed. using last iteration values
            warnings.warn("NLP solver not converged.")
            t0 = time.time()
            x, P = μ0, Σ0
            x_array.append(x)
            P_array.append(P)
            p = cs.vec(P)
            Nt = (x.shape[0] - 2)//2
            for tt in range(self.N):
                l_array.append(self.opti.debug.value(self.U[:, tt]))
                x_new = self.f(x, self.opti.debug.value(self.U[:, tt]))
                p = self.cov_trans(x, self.opti.debug.value(self.U[:, tt]), v, p)
                x = x_new
                x_array.append(x.full().flatten())
                P_array.append(cs.reshape(p, 2*Nt, 2*Nt).full())
            for tt in reversed(range(self.N)):
                L, S = self.lqr_backward(x_array[tt], l_array[tt], S)
                L_array.insert(0, L.full())
            elapsed += time.time() - t0
        elapsed += self.opti.stats()['t_proc_total']
        return x_array, P_array, l_array, L_array, elapsed

    @staticmethod
    def get_state_trans_func(x, u, dt):
        pos, vel = x[:2], u
        pos_new = pos + vel*dt
        x_new = cs.vcat([pos_new, x[2:]])
        f = cs.Function('f', [x, u], [x_new], ['x', 'u'], ['x_new'])
        return f

    @staticmethod
    def get_obs_func(x, v):
        pos = x[:2]
        Nt = (x.shape[0] - 2)//2
        y = cs.MX.zeros(Nt)
        for ii in range(Nt):
            pos_target_ii = x[2*(ii+1):2*(ii+2)]
            noise_ii = v[2*ii:2*(ii+1)]
            y[ii] = cs.sqrt(cs.sum1((pos - pos_target_ii - noise_ii)**2))
        h = cs.Function('h', [x, v], [y], ['x', 'v'], ['y'])
        return h

    @staticmethod
    def get_obs_cov_func(x):
        pos = x[:2]
        Nt = (x.shape[0] - 2)//2
        R_nominal = 0.001*cs.MX.eye(2*Nt)
        dists = cs.MX.zeros(2*Nt)
        for ii in range(Nt):
            pos_target_ii = x[2*(ii+1):2*(ii+2)]
            dist_ii = cs.sqrt(cs.sum1((pos - pos_target_ii)**2))
            dists[2*ii] = dist_ii
            dists[2*ii+1] = dist_ii
        R_scale = 0.01*cs.diag(dists)
        R = R_nominal + R_scale
        obs_cov = cs.Function('obs_cov', [x], [R], ['x'], ['R'])
        return obs_cov

    @staticmethod
    def get_obs_jacobians_func(h, x ,v):
        H = cs.jacobian(h(x, v), x)[:, 2:]
        M = cs.jacobian(h(x, v), v)
        obs_jacobians = cs.Function('obs_jacobians', [x, v], [H, M], ['x', 'v'], ['H', 'M'])
        return obs_jacobians

    @staticmethod
    def get_cov_trans_func(f, h, obs_jacobians, x, u, v, p, Q, obs_cov):
        Nt = (x.shape[0] - 2)//2
        P = cs.reshape(p, 2*Nt, 2*Nt)
        A = cs.jacobian(f(x, u), x)[2:,2:]
        x_new = f(x, u)
        H, M = obs_jacobians(x_new, v)
        P_pred = A@P@(A.T) + Q
        R = obs_cov(f(x, u))
        S = H@P_pred@(H.T) + M@R@(M.T)
        K = cs.mrdivide(P_pred@(H.T), S)
        P_updated = (cs.MX.eye(2*Nt) - K@H)@P_pred
        P_updated = (P_updated + P_updated.T)/2
        p_updated = cs.vec(P_updated)
        cov_trans = cs.Function('cov_trans', [x, u, v, p], [p_updated], ['x', 'u', 'v', 'p'], ['p_updated'])
        return cov_trans

    @staticmethod
    def get_stage_cost_func(u, Cu, dt):
        cost = 0.5*cs.dot(u, Cu@u)
        cost *= dt
        c_stage = cs.Function('c_stage', [u], [cost], ['u'], ['cost'])
        return c_stage

    @staticmethod
    def get_term_cost_func(x, p):
        Nt = (x.shape[0] - 2)//2
        P = cs.reshape(p, 2*Nt, 2*Nt)
        cost = 0.0
        for ii in range(Nt):
            Σ_ii = P[2*ii:2*(ii+1), 2*ii:2*(ii+1)]
            det_val = Σ_ii[0, 0]*Σ_ii[1, 1] - Σ_ii[1, 0]*Σ_ii[0, 1] # determinant of 2x2 matrix
            cost += cs.sqrt(((2*pi*e)**2)*det_val)
        c_term = cs.Function('c_term', [p], [cost], ['p'], ['cost'])
        return c_term

    @staticmethod
    def get_c_term_grad_func(c_term, p):
        g = cs.jacobian(c_term(p), p)
        c_term_grad = cs.Function('c_term_grad', [p], [g], ['p'], ['g'])
        return c_term_grad

    @staticmethod
    def gradient_control(cov_trans, c_term_grad, x, p):
        v = cs.MX.zeros(x.shape[0]-2)
        J = cs.jacobian(cov_trans(x, cs.MX.zeros(2), v, p), x)[:, :2]
        grad = J.T@c_term_grad(cov_trans(x, cs.MX.zeros(2), v, p)).T
        u = -1.0*grad/(cs.sqrt(cs.sum1(grad**2)) + 1e-10)
        gradient_control = cs.Function('gradient_control', [x, p], [u], ['x', 'p'], ['u'])
        return gradient_control

    @staticmethod
    def get_lqr_backward_func(f, x, u, S, Cs, Cu):
        A = cs.jacobian(f(x, u), x)
        B = cs.jacobian(f(x, u), u)
        L = cs.mldivide(Cu + (B.T)@S@B, (B.T)@S@A)
        S_prev = (A.T)@S@A - (A.T)@S@B@L + Cs
        S_prev = (S_prev + S_prev.T)/2
        lqr_backward = cs.Function('lqr_backward', [x, u, S], [L, S_prev], ['x', 'u', 'S'], ['L', 'S_prev'])
        return lqr_backward


class TLQG_Manipulate2D:
    def __init__(self, N, dto, Q, R, Cs, Cu, x_target, \
                 u_param_min, u_param_max, pos_gain, rot_gain):
        # parameters
        self.N = N # plan_horizon == N*dto
        self.dto = dto
        self.Q = Q
        self.R = R
        self.Cs = Cs # Wx in T-LQG paper
        self.Cu = Cu # Wu in T-LQG paper
        self.x_target = x_target
        self.u_param_min = u_param_min
        self.u_param_max = u_param_max
        self.pos_gain = pos_gain
        self.rot_gain = rot_gain

        # optimization variables and objective
        self.opti = None
        self.U = None
        self.cost = None

        # placeholder symbols
        self._x = cs.MX.sym('x', 11) # [px, py, θ, vx, vy, ω, m, J, rx, ry, μ]
        self._x_new = cs.MX.sym('x_new', 11)
        self._u = cs.MX.sym('u', 3)  # [fx, fy, tr]
        self._p = cs.MX.sym('p', 11**2) # vectorized covariance matrix
        self._S = cs.MX.sym('S', 11, 11) # LQR riccati matrix

        # function objects
        self.p_control = TLQG_Manipulate2D.p_control(self._x, self.pos_gain, self.rot_gain)
        self.f = TLQG_Manipulate2D.get_state_trans_func(self._x, self._u, self.dto)
        self.h = TLQG_Manipulate2D.get_obs_func(self._x, self._u)
        self.cov_trans = TLQG_Manipulate2D.get_cov_trans_func(self.f, self.h, self._x, self._x_new, self._u, self._p, self.Q, self.R)
        self.c_stage = TLQG_Manipulate2D.get_stage_cost_func(self._x, self._u, self._p, self.Cs, self.Cu, self.x_target, self.dto)
        self.c_term = TLQG_Manipulate2D.get_term_cost_func(self._x, self._p, self.Cs, self.x_target)
        self.lqr_backward = TLQG_Manipulate2D.get_lqr_backward_func(self.f, self._x, self._u, self._S, self.Cs, self.Cu)

    def setup_nlp(self, μ0, Σ0):
        # initialize variables
        self.opti = cs.Opti()
        self.U = self.opti.variable(3, self.N)
        self.opti.set_initial(self.U, 0)
        p = cs.vec(Σ0)
        x = μ0
        x_initial_guess = μ0
        self.cost = 0.0
        for tt in range(self.N):
            u_initial_guess = self.p_control(x_initial_guess)
            self.opti.set_initial(self.U[:, tt], u_initial_guess)
            x_initial_guess = self.f(x_initial_guess, u_initial_guess)
            x_new = self.f(x, self.U[:, tt])
            p = self.cov_trans(x, x_new, self.U[:, tt], p)
            x = x_new
            self.cost += self.c_stage(x, self.U[:, tt], p)
        self.cost += self.c_term(x, p)
        # set up minimization problem
        self.opti.minimize(self.cost)
        # control constraints
        self.opti.subject_to(self.opti.bounded(self.u_param_min[0], self.U[0, :], self.u_param_max[0]))
        self.opti.subject_to(self.opti.bounded(self.u_param_min[1], self.U[1, :], self.u_param_max[1]))
        self.opti.subject_to(self.opti.bounded(self.u_param_min[2], self.U[2, :], self.u_param_max[2]))

    def clear_opti(self):
        self.opti = None
        self.U = None
        self.cost = None

    def solve_tlqg(self, μ0, Σ0, verbose=True, online=True):
        t0 = time.time()
        self.clear_opti()
        self.setup_nlp(μ0, Σ0)
        if not verbose:
            if online: # we care about computation time of the entire T-LQG planning
                solver_options = dict([('print_level', 0), ('max_iter', 50), ('max_cpu_time', 0.2)])
            else:
                solver_options = dict([('print_level', 0), ('max_iter', 3000000), ('max_cpu_time', 3600)])

        else:
            if online:
                solver_options = dict([('max_iter', 50), ('max_cpu_time', 0.2)])
            else:
                solver_options = dict([('max_iter', 3000000), ('max_cpu_time', 3600)])
        self.opti.solver('ipopt', dict(), solver_options)
        x_array = []
        P_array = []
        l_array = []
        L_array = []
        S = self.Cs
        elapsed = time.time() - t0
        try:
            sol = self.opti.solve()
            t0 = time.time()
            x, P = μ0, Σ0
            x_array.append(x)
            P_array.append(P)
            p = cs.vec(P)
            for tt in range(self.N):
                l_array.append(sol.value(self.U[:, tt]))
                x_new = self.f(x, sol.value(self.U[:, tt]))
                p = self.cov_trans(x, x_new, sol.value(self.U[:, tt]), p)
                x = x_new
                x_array.append(x.full().flatten())
                P_array.append(cs.reshape(p, 11, 11).full())
            for tt in reversed(range(self.N)):
                L, S = self.lqr_backward(x_array[tt], l_array[tt], S)
                L_array.insert(0, L.full())
            elapsed += time.time() - t0
        except:
            # solve not finished within time budget or failed. using last iteration values
            warnings.warn("NLP solver not converged.")
            t0 = time.time()
            x, P = μ0, Σ0
            x_array.append(x)
            P_array.append(P)
            p = cs.vec(P)
            for tt in range(self.N):
                l_array.append(self.opti.debug.value(self.U[:, tt]))
                x_new = self.f(x, self.opti.debug.value(self.U[:, tt]))
                p = self.cov_trans(x, x_new, self.opti.debug.value(self.U[:, tt]), p)
                x = x_new
                x_array.append(x.full().flatten())
                P_array.append(cs.reshape(p, 11, 11).full())
            for tt in reversed(range(self.N)):
                L, S = self.lqr_backward(x_array[tt], l_array[tt], S)
                L_array.insert(0, L.full())
            elapsed += time.time() - t0
        elapsed += self.opti.stats()['t_proc_total']
        return x_array, P_array, l_array, L_array, elapsed

    @staticmethod
    def p_control(x, pos_gain, rot_gain):
        fVec = -pos_gain*x[:2];
        torque = -rot_gain*(x[2] - pi); # Target θ is pi.
        u = cs.vcat([fVec,torque])
        p_control = cs.Function('p_control', [x], [u], ['x'], ['u'])
        return p_control

    @staticmethod
    def get_state_trans_func(x, u, dt):
        pos,θ,vel,ω,m,J,r,μ = x[0:2],x[2],x[3:5],x[5],x[6],x[7],x[8:10],x[10]
        fx,fy,tr = u[0],u[1],u[2]
        pos_new = pos + vel*dt; # position of the center of mass.
        θ_new   = θ + ω*dt;     # angular velocity of the center of mass
        #θ_new = cs.mod(θ_new, 2*pi)
        vel_new = vel - μ/m*vel*dt + 1/m*cs.vcat([fx,fy])*dt;
        ω_new = ω - 1/J*cs.dot(r,cs.vcat([cs.sin(θ),cs.cos(θ)]))*fx*dt + \
                    1/J*cs.dot(r,cs.vcat([cs.cos(θ),-cs.sin(θ)]))*fy*dt + 1/J*tr*dt;
        x_new = cs.vcat([pos_new,θ_new,vel_new,ω_new,m,J,r,μ])
        f = cs.Function('f', [x, u], [x_new],['x', 'u'], ['x_new'])
        return f

    @staticmethod
    def get_obs_func(x, u):
        pos,θ,vel,ω,m,J,r,μ = x[0:2],x[2],x[3:5],x[5],x[6],x[7],x[8:10],x[10]
        fx,fy,tr = u[0],u[1],u[2]
        px_robot = pos[0] + r[0]*cs.cos(θ) - r[1]*cs.sin(θ)
        py_robot = pos[1] + r[0]*cs.sin(θ) + r[1]*cs.cos(θ)
        vx_robot = vel[0] - ω*(r[0]*cs.sin(θ) + r[1]*cs.cos(θ))
        vy_robot = vel[1] + ω*(r[0]*cs.cos(θ) - r[1]*cs.sin(θ))
        α_robot = 1/J*tr \
                - 1/J*(r[0]*cs.sin(θ) + r[1]*cs.cos(θ))*fx \
                + 1/J*(r[0]*cs.cos(θ) - r[1]*cs.sin(θ))*fy
        ax_robot = 1/m*(fx - μ*vel[0]) \
                 - α_robot*(r[0]*cs.sin(θ) + r[1]*cs.cos(θ)) \
                 - (ω**2)*(r[0]*cs.cos(θ) - r[1]*cs.sin(θ))
        ay_robot = 1/m*(fy - μ*vel[1]) \
                 + α_robot*(r[0]*cs.cos(θ) - r[1]*cs.sin(θ)) \
                 - (ω**2)*(r[0]*cs.sin(θ) + r[1]*cs.cos(θ))
        y = cs.vcat([px_robot,py_robot,θ,vx_robot,vy_robot,ω,ax_robot,ay_robot,α_robot])
        h = cs.Function('h', [x, u], [y], ['x', 'u'], ['y'])
        return h

    @staticmethod
    def get_cov_trans_func(f, h, x, x_new, u, p, Q, R):
        P = cs.reshape(p, 11, 11)
        A = cs.jacobian(f(x, u), x)
        H = cs.jacobian(h(x_new, u), x_new)
        P_pred = A@P@(A.T) + Q
        S = H@P_pred@(H.T) + R
        K = cs.mrdivide(P_pred@(H.T), S)
        P_updated = (cs.MX.eye(11) - K@H)@P_pred
        P_updated = (P_updated + P_updated.T)/2
        p_updated = cs.vec(P_updated)
        cov_trans = cs.Function('cov_trans', [x, x_new, u, p], [p_updated], ['x', 'x_new', 'u', 'p'], ['p_updated'])
        return cov_trans

    @staticmethod
    def get_stage_cost_func(x, u, p, Cs, Cu, x_target, dt):
        P = cs.reshape(p, 11, 11)
        cost = 0.5*cs.trace(P@Cs)
        cost += 0.5*cs.dot(x[:6] - x_target, Cs[:6, :6]@(x[:6] - x_target))
        cost += 0.5*cs.dot(u, Cu@u)
        cost *= dt
        c_stage = cs.Function('c_stage', [x, u, p], [cost], ['x', 'u', 'p'], ['cost'])
        return c_stage

    @staticmethod
    def get_term_cost_func(x, p, Cs, x_target):
        P = cs.reshape(p, 11, 11)
        cost = 0.5*cs.trace(P@Cs)
        cost += 0.5*cs.dot(x[:6] - x_target, Cs[:6, :6]@(x[:6] - x_target))
        c_term = cs.Function('c_term', [x, p], [cost], ['x', 'p'], ['cost'])
        return c_term

    @staticmethod
    def get_lqr_backward_func(f, x, u, S, Cs, Cu):
        A = cs.jacobian(f(x, u), x)
        B = cs.jacobian(f(x, u), u)
        L = cs.mldivide(Cu + (B.T)@S@B, (B.T)@S@A)
        S_prev = (A.T)@S@A - (A.T)@S@B@L + Cs
        S_prev = (S_prev + S_prev.T)/2
        lqr_backward = cs.Function('lqr_backward', [x, u, S], [L, S_prev], ['x', 'u', 'S'], ['L', 'S_prev'])
        return lqr_backward
