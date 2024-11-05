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

def solution_methods(v_initial, angle, t, method, air_res=0, gravity=g, mass=mass_ball, d=d_ball, rho=rho_air, h=h_initial):

    theta = angle
    tau = t
    vec = abs(v) #this, combined with rc make up the vector and value of v in the acceleration formula, dvdt, given
    rc = np.sqrt((v[0]**2) + (v[1])**2)
    A = (np.pi)*((d/2)**2)
    acc = np.array([0, gravity])

    a = -(acc) - (vec * (air_res * rho * A * rc)/(2*mass))
    v = np.array([v_initial * np.cos(theta), v_initial * np.sin(theta)]) #v has x and y components determined by trig
    r = np.array([0, h_initial])

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