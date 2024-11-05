#testing for commit
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

mass_ball = 0.145 #kg
d_ball = 7.4 #cm
g = 9.81 #m/s2
rho_air = 1.2 #kg/m3
C_d = 0.35 #drag coefficient
h_initial = 1 #m

#Part 1

def solution_methods(v, angle, t, method, air_res=0, gravity=g, mass=mass_ball, d=d_ball, rho=rho_air, drag=C_d, h=h_initial):
    if method == 'Euler':
        v_step = v + (tau * a)
        r_step = r + (tau * v)
        v, r = v_step, r_step

    elif method == 'Euler-Cromer':
        v_step = v + (tau * a)
        r_step = r + (tau * v_step)
        v, r = v_step, r_step

    elif method == 'Midpoint':
        v_step = v + (tau * a)
        r_step = r + (tau * ((v_step + v)/2))
        v, r = v_step, r_step