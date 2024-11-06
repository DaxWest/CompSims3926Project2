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

#Part 1

def solution_methods(v_initial, angle, t_step, method, air_res=0, gravity=g, mass=mass_ball, d=d_ball, rho=rho_air, h=h_initial):

    theta = angle
    tau = t_step

    r = np.array([0, h])
    v = np.array([v_initial * np.cos(theta), v_initial * np.sin(theta)])  # v has x and y components determined by trig

    #vec = abs(v) #this, combined with rc make up the vector and value of v in the acceleration formula, dvdt, given

    A = (np.pi)*((d/2)**2) #area
    acc = np.array([0, gravity])

    position = [r]
    while r[1] >= 0:
        rc = np.sqrt((v[0] ** 2) + (v[1]) ** 2)
        a = -(acc) - (abs(v) * (air_res * rho * A * rc) / (2 * mass))
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

        elif method == 'Theory':
            v_step = v + (tau * (-acc))
            r_step = r + (tau * v)
            v, r = v_step, r_step

        position = np.append(position, r)

    return position


angle = 45 * rads
v0 = 50 #m/s
t_step = 0.1

val_euler = solution_methods(v0, angle, t_step, 'Euler', air_res=C_d)
x_val_euler = val_euler[0::2]
y_val_euler = val_euler[1::2]

val_euler_cromer = solution_methods(v0, angle, t_step, 'Euler-Cromer', air_res=C_d)
x_val_euler_cromer = val_euler_cromer[0::2]
y_val_euler_cromer = val_euler_cromer[1::2]

val_midpoint = solution_methods(v0, angle, t_step, 'Midpoint', air_res=C_d)
x_val_midpoint = val_midpoint[0::2]
y_val_midpoint = val_midpoint[1::2]

val_theory = solution_methods(v0, angle, t_step, 'Theory', air_res=C_d)
x_val_theory = val_theory[0::2]
y_val_theory = val_theory[1::2]

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
ax[0].scatter(x_val_euler, y_val_euler, marker='+')
ax[0].set_title('Euler')

ax[1].scatter(x_val_euler_cromer, y_val_euler_cromer, marker='+')
ax[1].set_title('Euler-Cromer')

ax[2].scatter(x_val_midpoint, y_val_midpoint, marker='+')
ax[2].set_title('Midpoint')

fig.suptitle("Figure 2.3 Reproduction for Multiple Methods")
for i in range(3):
    ax[i].plot(x_val_theory, y_val_theory, 'k')
    ax[i].grid()
    ax[i].legend(['Drag = 0.35', 'Drag = 0'])

plt.savefig("Method_Comparison")

#Part 2
N=10
mean_speed = 100/2.237 #converting mph to m/s
std_speed = 15/2.237 #converting mph to m/s

mean_angle = 45*rads
std_angle = 10*rads

norm_dist_speed = (std_speed * np.random.randn(N)) + mean_speed
norm_dist_angle = (std_angle * np.random.randn(N)) + mean_angle

print(norm_dist_speed)
#RDH_sim = solution_methods(norm_dist_speed, norm_dist_angle, t_step, 'Euler', air_res=C_d)
