#testing for commit
import numpy as np
import matplotlib.pyplot as plt

mass_ball = 0.145 #kg
d_ball = 7.4 * (10**(-2)) #was given in cm, converted to m
g = 9.81 #m/s2
rho_air = 1.2 #kg/m3
C_d = 0.35 #drag coefficient
h_initial = 1 #m
rads = (np.pi)/180

#Part 1
def solution_methods(v_initial, angle, t_step, method, air_res=0, gravity=g, mass=mass_ball, d=d_ball, rho=rho_air, h=h_initial):
    '''

    :param v_initial: initial speed at which the ball is moving (in m/s)
    :param angle: initial angle from the horizontal (in radians)
    :param t_step: interval between when the position is "checked" (in s)
    :param method: accepts 'Euler', 'Euler-Cromer', 'Midpoint' or, 'Theory'. Is case-sensitive
    :param air_res: drag coefficient, 0 by default
    :param gravity: local value of acceleration due to gravity (in m/s^2)
    :param mass: mass of the object in motion (in kg)
    :param d: diameter of the object, assumed to be spherical/circular (in m)
    :param rho: mass density of the object (kg/m^3)
    :param h: initial height from which the object begins moving (in m)
    :return: returns the position (x,y) of the object

    '''
    theta = angle
    tau = t_step

    r = np.array([0, h]) #initial x and y position of the ball
    v = np.array([v_initial * np.cos(theta), v_initial * np.sin(theta)])  # v has x and y components determined by trig

    A = (np.pi)*((d/2)**2) #area
    acc = np.array([0, gravity])

    position = [r] #initializing position "vector"
    while r[1] >= 0:
        rc = np.sqrt((v[0] ** 2) + (v[1]) ** 2) #vector components of motion
        a = -(acc) - (abs(v) * (air_res * rho * A * rc) / (2 * mass)) #acceleration of the ball

        if method == 'Euler': #Euler method of time-step solutions for position and velocity of an object with or without air resistance
            v_step = v + (tau * a)
            r_step = r + (tau * v)
            v, r = v_step, r_step

        elif method == 'Euler-Cromer': #Euler-Cromer method of time-step solutions for position and velocity of an object with or without air resistance
            v_step = v + (tau * a)
            r_step = r + (tau * v_step)
            v, r = v_step, r_step

        elif method == 'Midpoint': #Midpoint method of time-step solutions for position and velocity of an object with or without air resistance
            v_step = v + (tau * a)
            r_step = r + (tau * ((v_step + v)/2))
            v, r = v_step, r_step

        elif method == 'Theory': #A theoretical method of time-step solutions for position and velocity of an object without air resistance, based on the Euler method
            v_step = v + (tau * (-acc))
            r_step = r + (tau * v)
            v, r = v_step, r_step

        position = np.append(position, r) #need to append both the original and new position so the correct information in carried into the next instance of the while loop

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
ax[0].set_xlabel('Range (m)')
ax[0].set_ylabel('Height (m)')

ax[1].scatter(x_val_euler_cromer, y_val_euler_cromer, marker='+')
ax[1].set_title('Euler-Cromer')
ax[1].set_xlabel('Range (m)')
ax[1].set_ylabel('Height (m)')

ax[2].scatter(x_val_midpoint, y_val_midpoint, marker='+')
ax[2].set_title('Midpoint')
ax[2].set_xlabel('Range (m)')
ax[2].set_ylabel('Height (m)')

fig.suptitle("Figure 2.3 Reproduction for Multiple Methods")
for i in range(3):
    ax[i].plot(x_val_theory, y_val_theory, 'k')
    ax[i].grid()
    ax[i].legend(['Drag = 0.35', 'Drag = 0'])

plt.savefig("Method_Comparison")

#Part 2
mean_speed = 100/2.237 #converting mph to m/s
std_speed = 15/2.237 #converting mph to m/s

mean_angle = 45*rads
std_angle = 10*rads

#chose to start with 100 bats
AB = 1000
HR = 0

for i in range(AB):
    norm_dist_speed = (std_speed * np.random.randn()) + mean_speed
    norm_dist_angle = (std_angle * np.random.randn()) + mean_angle
    RDH_sim = solution_methods(norm_dist_speed, norm_dist_angle, t_step, 'Euler', air_res=C_d)
    if (RDH_sim[0::2][-2] * 3.281) >= 400:
        HR += 1
print("------------------------------- Part 2 -------------------------------")
print(f'The RDH At-Bat to Homerun ratio: {AB / HR}')

#Part 3
h_fence = np.linspace(0.5, 15, 30)
HR_w_fence = np.zeros(30)

for i in range(AB):
    norm_dist_speed = (std_speed * np.random.randn()) + mean_speed
    norm_dist_angle = (std_angle * np.random.randn()) + mean_angle

    for val in h_fence:
        height_at_fence = 0
        RDH_sim_fence = solution_methods(norm_dist_speed, norm_dist_angle, t_step, 'Euler', air_res=C_d)
        range_sol = RDH_sim_fence[0::2]
        height_sol = RDH_sim_fence[1::2]

        index = np.array(np.where(range_sol * 3.281 >= 400)[0])
        if len(index) != 0:
            height_at_fence = height_sol[index[0]]

            if height_at_fence >= float(val):
                index_height = np.searchsorted(h_fence, val)
                HR_w_fence[index_height] = HR_w_fence[index_height] + 1
ratio_w_fence = AB/HR_w_fence
print("------------------------------- Part 3 -------------------------------")
abhr_leq_ten = []
abhr_gr_ten = []

for i in range(len(ratio_w_fence)):
    if ratio_w_fence[i] <= 10:
        abhr_leq_ten.append((ratio_w_fence[i], h_fence[i]))
    else:
        abhr_gr_ten.append((ratio_w_fence[i], h_fence[i]))

print(f'The minimum fence height (for the given conditions) to keep AB/HR > 10 is: {abhr_gr_ten[0][1]}m')