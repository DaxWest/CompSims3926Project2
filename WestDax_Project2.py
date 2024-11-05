#testing for commit
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

mass_ball = 0.145 #kg
d_ball = 7.4 * (10**(-2)) #was given in cm, converted to m
g = 9.81 #m/s2
rho_air = 1.2 #kg/m3
C_d = 0.35 #drag coefficient
h_initial = 1 #m
rads = (np.pi)/180
angle = 45 * rads

#Part 1

def solution_methods(v_initial, angle, t_step, method, air_res=0, gravity=g, mass=mass_ball, d=d_ball, rho=rho_air, h=h_initial):

    theta = angle
    tau = t_step
    vec = abs(v) #this, combined with rc make up the vector and value of v in the acceleration formula, dvdt, given
    rc = np.sqrt((v[0]**2) + (v[1])**2)
    A = (np.pi)*((d/2)**2) #area
    acc = np.array([0, gravity])

    a = -(acc) - (vec * (air_res * rho * A * rc)/(2*mass))
    v = np.array([v_initial * np.cos(theta), v_initial * np.sin(theta)]) #v has x and y components determined by trig
    r = np.array([0, h_initial])

    position = [r]
    while r[1] > 0:
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

        position = np.append(r)

    return position
v0 = 100 / 2.237 # 100 mph was given, have to convert mph to m/s
t_step = 0.1
val_euler = solution_methods(v0, angle, t_step, 'Euler', air_res=C_d)

fig = plt.figure()

#Part 2