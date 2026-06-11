import numpy as np
from pysph.solver.application import Application
from pysph.solver.solver import Solver

from pysph.sph.integrator import PECIntegrator
from pysph.sph.integrator_step import WCSPHStep

from pysph.sph.equation import Group
from pysph.sph.equation import Equation

from pysph.base.utils import get_particle_array

from compyle.api import declare
from math import pi

M_1_PI = 1.0 / pi


class TwoKernel(object):
    r"""
    WendlandQuintic is used for the inner layer; CubicSpline is used for the outer layer
    The subscript _BR corresponds to the outer layer
    """
    def __init__(self, dim=1):
        self.radius_scale = 1.5
        self.dim = dim

        # CubicSpline
        self.fac_cs = M_1_PI * 8.0

        # WendlandQuintic
        self.fac_wq = M_1_PI * 21.0 / 2.0

    # WendlandQuintic===========================================================
    def kernel(self, xij=[0., 0, 0], rij=1.0, h=1.0):

        # h : repulsive Radius
        h1 = 1.0 / h
        q = rij * h1

        fac = self.fac_wq * h1 * h1 * h1

        val = 0.0
        tmp = 1. - q
        if (q < 1.0):
            val = tmp * tmp * tmp * tmp * (4.0 * q + 1.0)

        return val * fac

    def dwdq(self, rij=1.0, h=1.0):
        h1 = 1.0 / h
        q = rij * h1

        # compute the gradient
        val = 0.0
        tmp = 1.0 - q
        if (q < 1.0):
            if (rij > 1e-12):
                val = -20.0 * q * tmp * tmp * tmp

        return val

    def gradient(self, xij=[0., 0, 0], rij=1.0, h=1.0,
                 grad=[0, 0, 0]):
        h1 = 1.0 / h
        # get the kernel normalizing factor
        fac = self.fac_wq * h1 * h1 * h1

        # compute the gradient.
        if (rij > 1e-12):
            wdash = self.dwdq(rij, h)
            tmp = fac * wdash * h1 / rij
        else:
            tmp = 0.0

        grad[0] = tmp * xij[0]
        grad[1] = tmp * xij[1]
        grad[2] = tmp * xij[2]

    # CubicSpline=========================================================================
    # for WIJBR-Update
    def kernel_BR(self, xij=[0., 0, 0], rij=1.0, H=1.0, W=[0]):
        # h=Ratt :cohesion Radius
        h1 = 1. / H
        q = rij * h1

        # get the kernel normalizing factor
        fac = self.fac_cs * h1 * h1 * h1

        tmp2 = 1. - q
        if (q > 1.0):
            val = 0.0
        elif (q > 0.5):
            val = 2. * tmp2 * tmp2 * tmp2
        else:
            val = 1 - 6. * q * q * (1 -  q)

        # return val * fac
        W[0] =val * fac

    # for DWIJbR-Update
    def dwdq_BR(self, rij=1.0, H=1.0):
        h1 = 1. / H
        q = rij * h1

        # compute dw_dq
        tmp2 = 1. - q
        if (rij > 1e-12):
            if (q > 1.0):
                val = 0.0
            elif (q > 0.5):
                val = -6. * tmp2 * tmp2
            else:
                val = 6. * q * (3. * q - 2)
        else:
            val = 0.0

        return val

    # for DWIJbR-Update
    def gradient_BR(self, xij=[0., 0, 0], rij=1.0, H=1.0, grad=[0, 0, 0]):
        h1 = 1. / H
        # get the kernel normalizing factor ( sigma )
        fac = self.fac_cs * h1 * h1 * h1
        # compute the gradient.
        if (rij > 1e-12):
            wdash = self.dwdq_BR(rij, H)
            tmp = fac * wdash * h1 / rij
        else:
            tmp = 0.0

        grad[0] = tmp * xij[0]
        grad[1] = tmp * xij[1]
        grad[2] = tmp * xij[2]


# Define the variables involved in the calculation and the variables to be saved.
def get_particle_array_wcsph(constants=None, **props):

    wcsph_props = ['cs', 'ax', 'ay', 'az', 'arho', 'x0', 'y0', 'z0', 'u0', 'v0', 'w0', 'rho0', 'div', 'dt_cfl','dt_force','apx', 'apy', 'apz','ahx', 'ahy', 'ahz','acx', 'acy', 'acz','wij']

    pa = get_particle_array(
        constants=constants, additional_props=wcsph_props, **props
    )

    # default property arrays to save out.
    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h', 'pid', 'gid', 'tag', 'p', 'apx', 'apy', 'apz','ahx', 'ahy', 'ahz','acx', 'acy', 'acz'
    ])

    return pa

class SolidWallPressureBC(Equation):
    r"""**Solid wall pressure boundary condition** [Adami2012]_

    This boundary condition is to be used with fixed ghost particles
    in SPH simulations and is formulated for the general case of
    moving boundaries.

    The velocity and pressure of the fluid particles is extrapolated
    to the ghost particles and these values are used in the equations
    of motion.

    Pressure boundary condition:

    The pressure of the ghost particle is also calculated from the
    fluid particle by interpolation using:

    .. math::

        p_g = \frac{\sum_f p_f W_{gf} + \boldsymbol{g - a_g} \cdot
        \sum_f \rho_f \boldsymbol{r}_{gf}W_{gf}}{\sum_f W_{gf}},

    where the subscripts `g` and `f` relate to the ghost and fluid
    particles respectively.

    Density of the wall particle is then set using this pressure

    .. math::

        \rho_w=\rho_0\left(\frac{p_w - \mathcal{X}}{p_0} +
        1\right)^{\frac{1}{\gamma}}
    """

    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        r"""
        Parameters
        ----------
        rho0 : float
            reference density
        p0 : float
            reference pressure
        b : float
            constant (default 1.0)
        gx : float
            Body force per unit mass along the x-axis
        gy : float
            Body force per unit mass along the y-axis
        gz : float
            Body force per unit mass along the z-axis

        Notes
        -----
        For a two fluid system (boundary, fluid), this equation must be
        instantiated with boundary as the destination and fluid as the
        source.

        The boundary particle array must additionally define a property
        :math:`wij` for the denominator in Eq. (27) from [Adami2012]. This
        array sums the kernel terms from the ghost particle to the fluid
        particle.
        """

        # self.rho0 = rho0
        # self.p0 = p0
        # self.b = b
        self.gx = gx
        self.gy = gy
        self.gz = gz

        super(SolidWallPressureBC, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_wij):
        d_p[d_idx] = 0.0
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, s_p, d_wij, s_rho, WIJ, XIJ):

        # numerator of Eq. (27) ax, ay and az are the prescribed wall
        # accelerations which must be defined for the wall boundary
        # particle
        gdotxij = self.gx*XIJ[0] + self.gy*XIJ[1] + self.gz*XIJ[2]

        d_p[d_idx] += s_p[s_idx]*WIJ + s_rho[s_idx]*gdotxij*WIJ

        # denominator of Eq. (27)
        d_wij[d_idx] += WIJ
    #
    def post_loop(self, d_idx, d_wij, d_p, d_rho):
        # extrapolated pressure at the ghost particle
        if d_wij[d_idx] > 1e-14:
            d_p[d_idx] /= d_wij[d_idx]

        d_p[d_idx]=max(0.0, d_p[d_idx])

    #     # update the density from the pressure Eq. (28)
    #     d_rho[d_idx] = self.rho0 * (d_p[d_idx]/self.p0 + self.b)


class TaitEOS(Equation):

    def __init__(self, dest, sources, rho0, c0, gamma, p0=0.0):
        # For pressure p_i=c^2 rho_0/ gamma ((rho_i/rho_0)^gamma-1)
        self.rho0 = rho0
        self.rho01 = 1.0 / rho0
        self.gamma = gamma
        self.c0 = c0
        self.B = rho0 * c0 * c0 / gamma
        self.p0 = p0

        # for Adaptive
        self.gamma1 = 0.5 * (gamma - 1.0)

        super(TaitEOS, self).__init__(dest, sources)

    def loop(self, d_idx, d_rho, d_p, d_cs):
        ratio = d_rho[d_idx] * self.rho01
        tmp = pow(ratio, self.gamma)
        d_p[d_idx] = self.p0 + self.B * (tmp - 1.0)

        d_cs[d_idx] = self.c0 * pow(ratio, self.gamma1)


class ContinuityEquationF(Equation):
    # d rho/d t= \sum_{j=1}^{N_f+N_s}m_j  u_{ij}\cdot\nabla_iW_{ij}^wq

    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, d_arho, s_idx, s_m, DWIJ, VIJ):
        vijdotdwij = DWIJ[0] * VIJ[0] + DWIJ[1] * VIJ[1] + DWIJ[2] * VIJ[2]
        d_arho[d_idx] += s_m[s_idx] * vijdotdwij


# Consider the momentum equation of negative pressure
class MomentumEquation_N(Equation):

    def __init__(self, dest, sources, theta=0.01):
        self.theta = theta  # Control the amount of negative pressure

        super(MomentumEquation_N, self).__init__(dest, sources)

    def initialize(self, d_idx, d_acx, d_acy, d_acz, d_dt_cfl):
        d_acx[d_idx] = 0.0
        d_acy[d_idx] = 0.0
        d_acz[d_idx] = 0.0
        d_dt_cfl[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_acx, d_acy, d_acz, s_m, XIJ, HIJ, SPH_KERNEL, RIJ, s_rho, d_rho):
        # 8gamma/M_1[W^cs] sum_{j=1}^{N_f} m_j/rho_i/rho_j \nabla_iW_{ij}^cs
        DWIJbR = declare('matrix(3)')

        # Consider negative pressure===================================================
        SPH_KERNEL.gradient_BR(XIJ, RIJ, 1.5*HIJ, DWIJbR)
        temp = self.theta / s_rho[s_idx] / d_rho[d_idx]

        d_acx[d_idx] += temp*s_m[s_idx] * DWIJbR[0]
        d_acy[d_idx] += temp*s_m[s_idx] * DWIJbR[1]
        d_acz[d_idx] += temp*s_m[s_idx] * DWIJbR[2]


class Adhesion(Equation):

    def __init__(self, dest, sources, alphaF=0.1, betaF=0):
        self.alphaF = alphaF  # att
        self.betaF = betaF  # att

        super(Adhesion, self).__init__(dest, sources)

    def initialize(self, d_idx, d_ahx, d_ahy, d_ahz):
        d_ahx[d_idx] = 0.0
        d_ahy[d_idx] = 0.0
        d_ahz[d_idx] = 0.0


    def loop(self, d_idx, s_idx, d_ahx, d_ahy, d_ahz,
             s_m, XIJ, HIJ, SPH_KERNEL, RIJ, s_rho, d_rho):
        DWIJbR = declare('matrix(3)')
        SPH_KERNEL.gradient_BR(XIJ, RIJ, 1.5*HIJ, DWIJbR)

        tem1 = self.alphaF / s_rho[s_idx] /d_rho[d_idx]
        d_ahx[d_idx] += tem1*s_m[s_idx] * DWIJbR[0]
        d_ahy[d_idx] += tem1*s_m[s_idx] * DWIJbR[1]
        d_ahz[d_idx] += tem1*s_m[s_idx] * DWIJbR[2]

        # # tem2= self.beta / s_rho[s_idx] / d_rho[d_idx]
        #
        # tem2 = self.betaF / s_rho[s_idx] / d_rho[d_idx]
        # d_ahx[d_idx] += -tem2*s_m[s_idx] * DWIJ[0]
        # d_ahy[d_idx] += -tem2*s_m[s_idx] * DWIJ[1]
        # d_ahz[d_idx] += -tem2*s_m[s_idx] * DWIJ[2]


class ME_Pressure(Equation):

    def __init__(self, dest, sources, c0, alpha=1.0, beta=1.0, tensile_correction=False):
        self.alpha = alpha
        self.beta = beta
        self.c0 = c0
        self.tensile_correction = tensile_correction

        super(ME_Pressure, self).__init__(dest, sources)

    def initialize(self, d_idx, d_apx, d_apy, d_apz):
        d_apx[d_idx] = 0.0
        d_apy[d_idx] = 0.0
        d_apz[d_idx] = 0.0


    def loop(self, d_idx, s_idx, d_rho, d_cs,
             d_p, s_m,
             s_rho, s_cs, s_p, VIJ,
             XIJ, HIJ, R2IJ, RHOIJ1, EPS,
             d_dt_cfl, DWIJ, d_apx, d_apy, d_apz):

        # Consider artificial viscosity===================================================
        vijdotxij = VIJ[0] * XIJ[0] + VIJ[1] * XIJ[1] + VIJ[2] * XIJ[2]

        piij = 0.0
        if vijdotxij < 0:
            cij = 0.5 * (d_cs[d_idx] + s_cs[s_idx])

            muij = (HIJ * vijdotxij) / (R2IJ + EPS)

            piij = -self.alpha * cij * muij + self.beta * muij * muij
            piij = piij * RHOIJ1

        # compute the CFL time step factor
        _dt_cfl = 0.0
        if R2IJ > 1e-12:
            _dt_cfl = abs(HIJ * vijdotxij / R2IJ) + self.c0
            d_dt_cfl[d_idx] = max(_dt_cfl, d_dt_cfl[d_idx])


        # Consider positive pressure==================================================
        rhoi21 = 1.0 / (d_rho[d_idx] * d_rho[d_idx])
        rhoj21 = 1.0 / (s_rho[s_idx] * s_rho[s_idx])

        tmp3 = (d_p[d_idx] * rhoi21 + s_p[s_idx] * rhoj21)

        d_apx[d_idx] += -s_m[s_idx] * (tmp3 + piij) * DWIJ[0]
        d_apy[d_idx] += -s_m[s_idx] * (tmp3 + piij) * DWIJ[1]
        d_apz[d_idx] += -s_m[s_idx] * (tmp3 + piij) * DWIJ[2]


class ME_Pressure2(Equation):

    def __init__(self, dest, sources, c0, alpha=1.0, beta=1.0, tensile_correction=False):
        self.alpha = alpha
        self.beta = beta
        self.c0 = c0
        self.tensile_correction = tensile_correction

        super(ME_Pressure2, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_rho, d_cs,
             d_p, s_m,
             s_rho, s_cs, s_p, VIJ,
             XIJ, HIJ, R2IJ, RHOIJ1, EPS,
             d_dt_cfl, DWIJ, d_apx, d_apy, d_apz):

        # DWIJmR = declare('matrix(3)')
        # SPH_KERNEL.gradient_BR(XIJ, RIJ, 1 * HIJ, DWIJmR)

        # Consider artificial viscosity===================================================
        vijdotxij = VIJ[0] * XIJ[0] + VIJ[1] * XIJ[1] + VIJ[2] * XIJ[2]

        piij = 0.0
        if vijdotxij < 0:
            cij = 0.5 * (d_cs[d_idx] + s_cs[s_idx])

            muij = (HIJ * vijdotxij) / (R2IJ + EPS)

            piij = -self.alpha * cij * muij + self.beta * muij * muij
            piij = piij * RHOIJ1

        # compute the CFL time step factor
        _dt_cfl = 0.0
        if R2IJ > 1e-12:
            _dt_cfl = abs(HIJ * vijdotxij / R2IJ) + self.c0
            d_dt_cfl[d_idx] = max(_dt_cfl, d_dt_cfl[d_idx])


        # Consider positive pressure==================================================
        rhoi21 = 1.0 / (d_rho[d_idx] * d_rho[d_idx])
        rhoj21 = 1.0 / (s_rho[s_idx] * s_rho[s_idx])

        tmp3 = (d_p[d_idx] * rhoi21 + s_p[s_idx] * rhoj21)

        d_apx[d_idx] += -s_m[s_idx] * (tmp3 + piij) * DWIJ[0]
        d_apy[d_idx] += -s_m[s_idx] * (tmp3 + piij) * DWIJ[1]
        d_apz[d_idx] += -s_m[s_idx] * (tmp3 + piij) * DWIJ[2]


class ME_ViscosityGravity(Equation):

    def __init__(self, dest, sources,
                 gx=0.0, gy=0.0, gz=0.0, mu=0.001, eta=0.01):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.mu = mu  # nu=mu/rho Viscosity coefficient
        # eta is the default parameter to prevent the calculation from crashing if the particles are too close
        self.eta = eta

        super(ME_ViscosityGravity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_au, d_av, d_aw, s_m,
             s_rho, VIJ, XIJ, HIJ, R2IJ, DWIJ):

        # Consider Viscosity============================================================
        murho = 2 * self.mu / (d_rho[d_idx] * d_rho[d_idx] + s_rho[s_idx] * s_rho[s_idx])

        # scalar part of the kernel gradient
        Fij = DWIJ[0] * XIJ[0] + DWIJ[1] * XIJ[1] + DWIJ[2] * XIJ[2]

        tmp2 = 4 * s_m[s_idx] * murho * Fij / (R2IJ + self.eta * HIJ * HIJ)

        # accelerations
        d_au[d_idx] += tmp2 * VIJ[0]
        d_av[d_idx] += tmp2 * VIJ[1]
        d_aw[d_idx] += tmp2 * VIJ[2]

    def post_loop(self, d_idx, d_au, d_av, d_aw, d_dt_force, d_apx, d_apy, d_apz,d_ahx, d_ahy, d_ahz, d_acx, d_acy, d_acz):
        d_au[d_idx] += self.gx +d_apx[d_idx]+d_ahx[d_idx]+d_acx[d_idx]
        d_av[d_idx] += self.gy +d_apy[d_idx]+d_ahy[d_idx]+d_acy[d_idx]
        d_aw[d_idx] += self.gz +d_apz[d_idx]+d_ahz[d_idx]+d_acz[d_idx]

        acc2 = (d_au[d_idx] * d_au[d_idx] +
                d_av[d_idx] * d_av[d_idx] +
                d_aw[d_idx] * d_aw[d_idx])

        # store the square of the max acceleration
        d_dt_force[d_idx] = acc2


class XSPHCorrection(Equation):

    def __init__(self, dest, sources, eps=0.5):
        self.eps = eps
        super(XSPHCorrection, self).__init__(dest, sources)

    def initialize(self, d_idx, d_ax, d_ay, d_az):
        d_ax[d_idx] = 0.0
        d_ay[d_idx] = 0.0
        d_az[d_idx] = 0.0

    def loop(self, s_idx, d_idx, s_m, d_ax, d_ay, d_az, WIJ, RHOIJ1, VIJ):
        tmp = -self.eps * s_m[s_idx] * WIJ * RHOIJ1

        d_ax[d_idx] += tmp * VIJ[0]
        d_ay[d_idx] += tmp * VIJ[1]
        d_az[d_idx] += tmp * VIJ[2]

    def post_loop(self, d_idx, d_ax, d_ay, d_az, d_u, d_v, d_w):
        d_ax[d_idx] += d_u[d_idx]
        d_ay[d_idx] += d_v[d_idx]
        d_az[d_idx] += d_w[d_idx]


def create_particles(fdx, fluid_column_R, V0, h, dim, rho0, rhoC0):
    m = (fdx ** dim) * rho0  # (kg)  

    xf, yf, zf = np.mgrid[-fluid_column_R:fluid_column_R:fdx,
                 - fluid_column_R:fluid_column_R:fdx,
                 - fluid_column_R:fluid_column_R:fdx]
    mask = xf * xf + yf * yf + zf * zf < fluid_column_R * fluid_column_R
    xf = xf[mask]
    yf = yf[mask]
    zf = zf[mask]
    zf += fluid_column_R + fdx * 40


    sdx = 0.8 * fdx
    Depth = h
    L_lim=0.2
    R_lim=2.4-L_lim/2

    alpha_lim=np.pi/5

    xb, yb, zb = np.mgrid[-2*(R_lim+3*L_lim):(R_lim+3*L_lim):sdx, -(R_lim+3*L_lim):(R_lim+3*L_lim):sdx,
                 -Depth:0:sdx]
    xt = xb.flatten()
    yt = yb.flatten()
    zt = zb.flatten()

    # mask = xt * xt + yt * yt + (zt - fluid_column_R / 2) * (
    #         zt - fluid_column_R / 2) <= (R_lim+3*L_lim) * (R_lim+3*L_lim)
    mask = (
            (-2 * (R_lim + 3 * L_lim) < xt) & (xt < (R_lim + 3 * L_lim)) &
            (-(R_lim + 3 * L_lim) < yt) & (yt < (R_lim + 3 * L_lim))
    )

    maskq1 = xt * xt + yt * yt + (zt - fluid_column_R / 2) * (
            zt - fluid_column_R / 2) <= (R_lim+L_lim) * (R_lim+L_lim)
    maskq2 = xt * xt + yt * yt + (zt - fluid_column_R / 2) * (
            zt - fluid_column_R / 2) >= R_lim * R_lim

    maskq3 = np.abs(yt)   <= np.tan(alpha_lim/2)*xt
    maskq4 = xt > 0

    maskq = maskq1 & maskq2 & ~(maskq3 & maskq4)
    xtq = xt[maskq]
    ytq = yt[maskq]
    ztq = zt[maskq]

    xt = xt[mask & ~maskq]
    yt = yt[mask & ~maskq]
    zt = zt[mask & ~maskq]

    m2 = sdx ** dim * rhoC0  # dim -1 is equivalent to dividing sdx

    boundary = get_particle_array_wcsph(name='boundary', x=xt, y=yt, z=zt, h=h, m=m2, rho=rhoC0)

    boundaryq = get_particle_array_wcsph(name='boundaryq', x=xtq, y=ytq, z=ztq, h=h, m=m2, rho=rhoC0)

    xb_new, yb_new, zb_new = np.mgrid[-(R_lim + 3 * L_lim):(R_lim + 3 * L_lim):fdx, -(R_lim + 3 * L_lim):(R_lim + 3 * L_lim):fdx,
    -Depth:0:fdx]
    xt_new = xb_new.flatten()
    yt_new = yb_new.flatten()
    zt_new = zb_new.flatten()
    maskq1 = xt_new * xt_new + yt_new * yt_new + (zt_new - fluid_column_R / 2) * (
            zt_new - fluid_column_R / 2) <= (R_lim + L_lim) * (R_lim + L_lim)
    maskq2 = xt_new * xt_new + yt_new * yt_new + (zt_new - fluid_column_R / 2) * (
            zt_new - fluid_column_R / 2) >= R_lim * R_lim

    maskq3 = np.abs(yt_new)  <= np.tan(alpha_lim / 2) *xt_new
    maskq4 = xt_new > 0

    maskq = maskq1 & maskq2 & ~(maskq3 & maskq4)

    xtq = xt_new[maskq]
    ytq = yt_new[maskq]
    ztq = zt_new[maskq]

    maskqf = ztq+2*fdx >0
    xtqf = xtq[maskqf]
    ytqf = ytq[maskqf]
    ztqf = ztq[maskqf]+0.12
    xf=np.concatenate([xf, xtqf.flatten()])
    yf = np.concatenate([yf, ytqf.flatten()])
    VV=(zf-zf+1)*(-V0)
    zf = np.concatenate([zf, ztqf.flatten()])
    VV2=ztqf.flatten()*0
    VVall = np.concatenate([VV, VV2])
    fluid = get_particle_array_wcsph(name='fluid', x=xf, y=yf, z=zf, h=h, m=m, rho=rho0, w=VVall)

    return [fluid, boundary, boundaryq]


def _get_post_process_props(array):
    """Return x, y, m, u, v, p.
    """
    x, y, z = array.get(
        'x', 'y', 'z',
    )
    return x, y, z


class Drop(Application):

    def __init__(self, We, A, B):
        self.ParameterA = A
        self.ParameterB = B
        self.We = We
        super(Drop, self).__init__()

    def consume_user_options(self):
        self.fdx = 0.06  # (mm) average particle spacing
        self.Ratt = 0.24  # support radius

        self.fluid_column_R = 1.1  # (mm)
        self.rho0 = 0.001  # (g/mm^3) fluid density
        self.rhoC0 = 0.001  # (g/mm^3) The density of the base layer
        self.V0 = np.sqrt(self.We * 0.0000720 / (self.rho0 * self.fluid_column_R))  # mm/ms
        self.c0 = 10.0 * (1 + self.V0)

        # 2*self.Parameter - self.ParameterB - 1 = cos(theta)

        M1CS = 31 * self.Ratt / 70
        A = 8 * 0.0000720 / M1CS
        self.theta = A * 1
        self.alphaF = A * self.ParameterA
        # M1ns = 3 * (self.Ratt / 1.0) / 8
        # B = 8 * 0.0000720 / M1ns
        M1wq = 5 * (self.Ratt / 1.5) / 12
        B = 8 * 0.0000720 / M1wq
        self.betaF = B * self.ParameterB

        self.tf = 1  # (ms)
        self.p0 = 0
        self.dim = 3
        self.mu = 0.000001  # (10^-6 g/(ms.mm) ~ 1 cP) dynamic viscosity
        self.alpha = 0.3  # produces shear and bulk viscosity
        self.beta = 0.0  # used to handle high Mach number shocks
        g = 0.00981  # (mm/ms^2) gravitational acceleration
        if self.dim == 2:
            self.gy = -g
            self.gz = 0
        else:
            self.gy = 0
            self.gz = -g
        self.gamma = 7  # compressibility coefficient; larger is better
        self.dt = 0.04 * self.Ratt / self.c0  # (s) prescribed time step
        print("self.ParameterA = %2.2f,  Ratt= %2.2f,  fdx= %4.4f" % (
            self.ParameterA, self.Ratt, self.fdx))

    def create_solver(self):
        kernel = TwoKernel(dim=self.dim)
        integrator = PECIntegrator(fluid=WCSPHStep())

        solver = Solver(kernel=kernel, dim=self.dim, integrator=integrator, dt=self.dt, tf=self.tf, pfreq=500)
        return solver

    def create_equations(self):
        equations = [
            Group(
                equations=[
                    TaitEOS(dest='fluid', sources=None, rho0=self.rho0, c0=self.c0, gamma=self.gamma, p0=self.p0),
                    SolidWallPressureBC(dest='boundary', sources=['fluid', 'boundaryq'], gx=0, gy=0, gz=self.gz),
                    SolidWallPressureBC(dest='boundaryq', sources=['fluid', 'boundary'], gx=0, gy=0, gz=self.gz),
                ], real=False
            ),
            Group(
                equations=[
                    ContinuityEquationF(dest='fluid', sources=['fluid', 'boundary', 'boundaryq']),

                    # Negative pressure (not present in fluid-solid interaction)
                    MomentumEquation_N(dest='fluid', sources=['fluid'], theta=self.theta),
                    # Adhesion force
                    Adhesion(dest='fluid', sources=['boundaryq'], alphaF=self.alphaF, betaF=0),
                    # Adhesion(dest='fluid', sources=['boundary'], alphaF=0, betaF=self.betaF),

                    # Pressure
                    ME_Pressure(dest='fluid', sources=['fluid', 'boundary', 'boundaryq'],
                                c0=self.c0, alpha=self.alpha, beta=self.beta, tensile_correction=False),
                    # Positive pressure
                    ME_ViscosityGravity(dest='fluid', sources=['fluid'], mu=self.mu, gx=0, gy=self.gy, gz=self.gz),

                    XSPHCorrection(dest='fluid', sources=['fluid']),
                ]
            ),
        ]

        # This is required since MomentumEquation (ME) adds artificial
        # viscosity (AV), so make alpha 0.0 for ME and enable delta sph AV.
        # alpha = 0.0 if self.scheme.scheme.delta_sph else self.scheme.scheme.alpha

        return equations

    def post_process(self, info_fname):
        info = self.read_info(info_fname)
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output
        import os

        file_path = self.output_dir + '/' + self.fname + '.txt'
        # Open the file in write mode
        with open(file_path, 'w') as file:
            file.write(
                "ParameterA = %3.3f, fdx= %3.3f, Ratt= %3.3f" % (
                    self.ParameterA, self.fdx, self.Ratt) + "\n"
                + "dt = %g, fluid_column_R = %2.2f, V = %3.3f" % (
                    self.dt, self.fluid_column_R, self.V0) + "\n"
            )

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['fluid']
        b.scalar = 'vmag'
        b.plot.actor.property.render_points_as_spheres=True
        particle_arrays['boundary'].plot.actor.property.opacity = 0.2
        from mayavi import mlab
        mlab.view(azimuth=180, elevation=0, distance='auto', focalpoint='auto')
        ''')

    def create_particles(self):
        V0 = np.sqrt(self.V0 * self.V0 + 2 * self.gz * 40 * self.fdx)
        return create_particles(self.fdx, self.fluid_column_R, V0, self.Ratt / 1.5, self.dim, self.rho0, self.rhoC0)


if __name__ == '__main__':  # Check whether the script is executed directly
    app = Drop(32.8, 0.966, 0)
    app.run()
    app.post_process(app.info_filename)