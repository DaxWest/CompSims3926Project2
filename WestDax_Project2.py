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

#initial conditions
angle = 45 * rads
v0 = 50 #m/s
t_step = 0.1

#solution for the movement of the ball by the Euler method
val_euler = solution_methods(v0, angle, t_step, 'Euler', air_res=C_d)
x_val_euler = val_euler[0::2]
y_val_euler = val_euler[1::2]

#solution for the movement of the ball by the Euler-Cromer method
val_euler_cromer = solution_methods(v0, angle, t_step, 'Euler-Cromer', air_res=C_d)
x_val_euler_cromer = val_euler_cromer[0::2]
y_val_euler_cromer = val_euler_cromer[1::2]

#solution for the movement of the ball by the Midpoint method
val_midpoint = solution_methods(v0, angle, t_step, 'Midpoint', air_res=C_d)
x_val_midpoint = val_midpoint[0::2]
y_val_midpoint = val_midpoint[1::2]

#theorectical solution for the movement of the ball by the Euler method without air resistance
val_theory = solution_methods(v0, angle, t_step, 'Theory', air_res=C_d)
x_val_theory = val_theory[0::2]
y_val_theory = val_theory[1::2]

#plotting the reproduction of figure 2.3 from the textbook
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
AB = 1000 #total batting instances
HR = 0 #initial homerun count

for i in range(AB):
    norm_dist_speed = (std_speed * np.random.randn()) + mean_speed #returns an array of values making up a normal distribution of speed values based on a mean and std value
    norm_dist_angle = (std_angle * np.random.randn()) + mean_angle #returns an array of values making up a normal distribution of angle values based on a mean and std value
    #chose to use the Euler method (arbitrary choice) to find the solution to the motion of the ball
    RDH_sim = solution_methods(norm_dist_speed, norm_dist_angle, t_step, 'Euler', air_res=C_d)

    #counting the number of homeruns
    if (RDH_sim[0::2][-2] * 3.281) >= 400: #an instance is considered to have been a homerun if its range (x dimension) is greater than or equal to 400 feet
        HR += 1

print("------------------------------- Part 2 -------------------------------")
#finding the ratio between the number of hits and the number of homeruns the automatic batter produces
print(f'The RDH At-Bat to Homerun ratio: {AB / HR}')

#Part 3
h_fence = np.linspace(0.5, 15, 30) #an array of values for the height of the ball diamond fence that increases in increments of 0.5m, from 0.5m to 15m high
HR_w_fence = np.zeros(30) #initial number of homeruns hit, corresponding to each fence height

#same number of batting attempts as part 2
for i in range(AB):
    norm_dist_speed = (std_speed * np.random.randn()) + mean_speed #returns an array of values making up a normal distribution of speed values based on a mean and std value
    norm_dist_angle = (std_angle * np.random.randn()) + mean_angle #returns an array of values making up a normal distribution of angle values based on a mean and std value

    #calculations are then performed for each fence height
    for val in h_fence:
        height_at_fence = 0 #initial condition assumes no homerun
        #chose to use the Euler method (arbitrary choice) to find the solution to the motion of the ball
        RDH_sim_fence = solution_methods(norm_dist_speed, norm_dist_angle, t_step, 'Euler', air_res=C_d)
        range_sol = RDH_sim_fence[0::2] #x axis position values
        height_sol = RDH_sim_fence[1::2] #y axis position values

        #checking where the range meets or exceeds the distance required to be considered a homerun
        index = np.array(np.where(range_sol * 3.281 >= 400)[0])
        #only considering the case where the batter makes at least one homerun
        if len(index) != 0:
            height_at_fence = height_sol[index[0]] #sets the value of the height of the ball at the fence to be equal to the y position value when the x position >= 400 feet

            if height_at_fence >= float(val): #the ball clears the fence and is considered for the homerun count if its height is greater than the current value for the fence height
                index_height = np.searchsorted(h_fence, val)
                HR_w_fence[index_height] = HR_w_fence[index_height] + 1

#calculates the ratio of batting instances to homeruns, accounting for fence height
ratio_w_fence = AB/HR_w_fence
print("------------------------------- Part 3 -------------------------------")
abhr_leq_ten = [] #all ratios less than or equal to 10
abhr_gr_ten = [] #all ratios greater than 10

#appends two lists with a tuple of the ratio and fence height depending on the value of the ratio wrt 10
for i in range(len(ratio_w_fence)):
    if ratio_w_fence[i] <= 10:
        abhr_leq_ten.append((ratio_w_fence[i], h_fence[i]))
    else:
        abhr_gr_ten.append((ratio_w_fence[i], h_fence[i]))

print(f'The minimum fence height (for the given conditions) to keep AB/HR > 10 is: {abhr_gr_ten[0][1]}m')