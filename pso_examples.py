import numpy as np
from pyswarm import pso

###############################################################################

print('*'*65)
print('Example minimization of 4th-order banana function (no constraints)')
def myfunc(x):
    x1 = x[0]
    x2 = x[1]
    return x1**4 - 2*x2*x1**2 + x2**2 + x1**2 - 2*x1 + 5

bounds = [(-3, 2), (-1, 6)]

xopt1, fopt1 = pso(myfunc, bounds)

print('The optimum is at:')
print('    {}'.format(xopt1))
print('Optimal function value:')
print('    myfunc: {}'.format(fopt1))

###############################################################################

print('*'*65)
print('Example minimization of 4th-order banana function (with constraint)')
def mycon(x):
    x1 = x[0]
    x2 = x[1]
    return [-(x1 + 0.25)**2 + 0.75*x2]

xopt2, fopt2 = pso(myfunc, bounds, f_ieqcons=mycon)

print('The optimum is at:')
print('    {}'.format(xopt2))
print('Optimal function value:')
print('    myfunc: {}'.format(fopt2))
print('    mycon : {}'.format(mycon(xopt2)))

###############################################################################

print('*'*65)
print('Engineering example: minimization of twobar truss weight, subject to')
print('  Yield Stress <= 100 kpsi')
print('  Yield Stress <= Buckling Stress')
print('  Deflection   <= 0.25 inches')
def weight(x, *args):
    H, d, t = x  # all in inches
    B, rho, E, P = args
    return rho*2*np.pi*d*t*np.sqrt((B/2)**2 + H**2)
    
def stress(x, *args):
    H, d, t = x  # all in inches
    B, rho, E, P = args
    return (P*np.sqrt((B/2)**2 + H**2))/(2*t*np.pi*d*H)

def buckling_stress(x, *args):
    H, d, t = x  # all in inches
    B, rho, E, P = args
    return (np.pi**2*E*(d**2 + t**2))/(8*((B/2)**2 + H**2))

def deflection(x, *args):
    H, d, t = x  # all in inches
    B, rho, E, P = args
    return (P*np.sqrt((B/2)**2 + H**2)**3)/(2*t*np.pi*d*H**2*E)

def mycons(x, *args):
    strs = stress(x, *args)
    buck = buckling_stress(x, *args)
    defl = deflection(x, *args)
    return [100 - strs, buck - strs, 0.25 - defl]

B = 60  # inches
rho = 0.3  # lb/in^3
E = 30000  # kpsi
P = 66  # lb (force)
args = (B, rho, E, P)
bounds = [(10, 30), (1, 3), (0.01, 0.25)]
xopt4, fopt4 = pso(weight, bounds, f_ieqcons=mycons, args=args)

print('The optimum is at:')
print('    {}'.format(xopt4))
print('Optimal function values:')
print('    weight         : {}'.format(fopt4))
print('Constraint functions:')
print('    stress         : {}'.format(stress(xopt4, *args)))
print('    buckling stress: {}'.format(buckling_stress(xopt4, *args)))
print('    deflection     : {}'.format(deflection(xopt4, *args)))


print('Example minimisation with non float parameters')
def myfunc(x):
    return np.sum(x**2)

bounds = [(-3, 3), (-3, 3), [-3, 3]]

def transform_pars(pars):
    return (
        # regular float
        pars[0],
        # only integers
        round(pars[1]),
        # only in the predefined list
        param_2_possibilities[np.argmin(np.abs(param_2_possibilities - pars[2]))]
    )

param_2_possibilities = np.array([
    -2, -1, -.1, .01, .2, .3, .35, 2, 3
])

xopt, fopt = pso(myfunc, bounds, transform_pars=transform_pars,
                        #debug=True, particle_output=True,
                        maxiter=100, swarmsize=100, processes=1)

print('The optimum is at:')
print('    {}'.format(xopt))
print('Optimal function value:')
print('    myfunc: {}'.format(fopt))