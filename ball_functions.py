import numpy as np
from scipy.integrate import odeint
from scipy import stats
from scipy.stats import uniform
import matplotlib.pyplot as plt
from matplotlib  import cm
import matplotlib as mpl
import math


def main():

	# Simulation parameters
	v0           = 1.0      # velocity
	eta          = 0.5      # random fluctuation in angle (in radians)
	L            = 10       # size of box
	R            = 1        # interaction radius
	dt           = 0.2      # time step
	Nt           = 200      # number of time steps
	N            = 6      # number of balls
	
	# Initialize
	np.random.seed(17)      # set the random number generator seed

	# initial positions
	x = np.random.rand(N)*L
	y = np.random.rand(N)*L
	
	# initial velocities
	theta = 2 * np.pi * np.random.rand(N)
	vx = v0 * np.cos(theta)
	vy = v0 * np.sin(theta)
	
    # Arrays to store positions and orientations
	x_time = np.array(np.zeros((Nt,N)))
	y_time = np.array(np.zeros((Nt,N)))
	theta_time = np.array(np.zeros((Nt,N)))
	x_time[0,:] = x

	# Simulation Main Loop
	for i in range(Nt):

		# move
		x += vx*dt
		y += vy*dt
		
		# apply periodic BCs
		x = x % L
		y = y % L
		
        #
		dists = (x-5)**2 + (y-5)**2
		
		# find mean angle of neighbors within R
		mean_theta = np.copy(theta)
		theta_rad = np.zeros(N)

		for b in range(N):
			neighbors = (x-x[b])**2+(y-y[b])**2 < R**2
			sx = np.sum(np.cos(theta[neighbors]))
			sy = np.sum(np.sin(theta[neighbors]))
			mean_theta[b] = np.arctan2(sy, sx)
	
			
		# add random perturbations
		theta = mean_theta + eta*(np.random.rand(N)-0.5)
		#theta -= theta_rad
		
		# update velocities
		vx = v0 * np.cos(theta)
		vy = v0 * np.sin(theta)

		# append
		x_time[i,:] = x
		y_time[i,:] = y
		theta_time[i,:] = theta
	
	return x_time, y_time, theta_time

def dynamic_gen():

    # Simulation parameters
	v0           = 100.0      # velocity
	eta          = 0.5      # random fluctuation in angle (in radians)
	L            = 100       # size of box
	R            = 5        # interaction radius
	dt           = 0.01      # time step
	N            = 2      # number of balls
      
    #
	x,y = np.random.uniform(low=20, high=80, size=(2, N))
	vx,vy = v0*np.random.uniform(low=1, high=1, size=(2, N))
	v0x,v0y = v0*np.random.choice([1,1], size=(2, N))
	theta = 2 * np.pi * np.random.rand(N)
	#print(theta)

	yield x,y,theta
	r = 10

	while True:

		#theta += 0.5*(np.random.rand(N)-0.5)

		
		dx = x[:,None]-x[None,:]
		dy = y[:,None]-y[None,:]
		
        #
		dr = np.hypot(dx,dy)
		dr = np.ma.masked_where( ~((0<dr)&(dr<r)), dr )
        
		#
		b2c = np.array(vectors_from_point([50,50], x, y))
		vec_pointing = np.array(np.column_stack((np.cos(theta), np.sin(theta))))
		ang_between = angle_between(b2c, vec_pointing, N)

		
		ang_centre = np.array([math.atan2(y[i]-50,x[i]-50) % (2*np.pi) for i in range(N)])

		
		diff = (theta-ang_centre)
		ori = np.sign(diff)
		#print(diff, np.pi-ang_between)


		cross_x = np.where(((np.sqrt((x-50)**2+(y-50)**2) > 50-r) & (ang_between >= np.pi/2)), 1*np.cos(ang_between), 1)
		cross_y = np.where(((np.sqrt((x-50)**2+(y-50)**2) > 50-r) & (ang_between >= np.pi/2)), 1*np.cos(ang_between), 1)
		cross_x = np.ones(N)
		cross_y = np.ones(N)
		theta = np.where(((np.sqrt((x-50)**2+(y-50)**2) > 50-r) & (ang_between >= np.pi/2)), theta - ori*(2*np.abs(diff)-np.pi), theta)

		### collisions ###
		#b2b = np.array(vectors_between_points(x, y))
		#vec_pointing = np.array(np.column_stack((np.cos(theta), np.sin(theta))))
		#ang_between = angles_between_vectors(b2b, vec_pointing)
            
		#print(ang_between*180/np.pi)
        #print(ve)

		#np.where(dr < 2*r + 1, ,0)



		x += cross_x*v0x*np.cos(theta)*dt
		y += cross_y*v0y*np.sin(theta)*dt

		yield x,y,theta
            
def angles_between_vectors(vectors1, vectors2):

    angles = []
    
    for vec1, vec2 in zip(vectors1, vectors2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        cosine_angle = dot_product / (norm_vec1 * norm_vec2)
        angle_rad = np.arccos(cosine_angle)
        angles.append(angle_rad)
    
    return np.array(angles)

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2, N):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angles = np.array([np.arccos(np.clip(np.dot(v1_u[i], v2_u[i].T), -1.0, 1.0)) for i in range(N)])
    return angles

def vectors_from_point(ref_point, x_positions, y_positions):
    # Create an array of vectors from ref_point to all other points
    vectors = np.array([-x_positions + ref_point[0], -y_positions + ref_point[1]]).T
    
    # Calculate the magnitudes of all vectors
    magnitudes = np.linalg.norm(vectors, axis=1)
    
    # Calculate the unit vectors
    unit_vectors = vectors / magnitudes[:, None]
    
    return unit_vectors

def vectors_between_points(xs, ys):

    num_points = len(xs)
    vectors = []
    
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                dx = xs[j] - xs[i]
                dy = ys[j] - ys[i]
                vectors.append((dx, dy))
    
    return vectors

def dynamic_gen_periodic():


    # Simulation parameters
	v0           = 3.0      # velocity
	eta          = 0.5      # random fluctuation in angle (in radians)
	L            = 100       # size of box
	R            = 5        # interaction radius
	dt           = 0.1      # time step
	N            = 50      # number of balls
	
	x,y = np.random.uniform(low=0, high=L, size=(2, N))
	v0x,v0y = np.random.uniform(low=-1, high=1, size=(2, N))
	#v0x,v0y = np.random.choice([-1,1], size=(2, N))
	
	yield x,y
	r = 5
	k = 10
	vx = v0x
	vy = v0y
	while True:

		x += vx*dt; x = x % L
		y += vy*dt; y = y % L
        # dx[i,j] = x[i]-x[j]
		dx = x[:,None]-x[None,:]; dx = dx % L
		dy = y[:,None]-y[None,:]; dy = dy % L
		
		dr = np.hypot(dx,dy)
		dr = np.ma.masked_where( ~((0<dr)&(dr<r)), dr)
        
        # size of force if less than or equal to radius
		force = k*(2*r-dr)
		vx = v0x + np.ma.sum(force*dx/dr,axis=1)*dt
		vy = v0x + np.ma.sum(force*dy/dr,axis=1)*dt



		yield x,y


def scatter_t(x,y,theta,t):
    
    fig = plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
    norm = mpl.colors.Normalize(vmin=0, vmax=2*np.pi)
    cmap = cm.gist_rainbow
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    temp = m.to_rgba(np.mod(theta[t,:],2*np.pi))
    plt.scatter(x[t,:],y[t,:], c = temp, s = 100, alpha = 0.9, marker = 'o',edgecolors='none',cmap = cm.gist_rainbow)
    plt.xlim(0,10)
    plt.ylim(0,10)

    #theta = np.linspace( 0 , 2 * np.pi , 150 )
 
    #radius = 100
    
    #a = radius * np.cos( theta )
    #b = radius * np.sin( theta )
    
    #plt.plot( a, b )
	
def scatter(x,y,theta):
    
    fig = plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
    norm = mpl.colors.Normalize(vmin=0, vmax=2*np.pi)
    cmap = cm.gist_rainbow
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    temp = m.to_rgba(np.mod(theta,2*np.pi))
    plt.scatter(x,y, c = temp, s = 10000, alpha = 0.9, marker = 'o',edgecolors='none',cmap = cm.gist_rainbow)
    plt.xlim(0,100)
    plt.ylim(0,100)
	

    b2c = np.array(vectors_from_point([50,50], x, y))
    vec_pointing = np.array(np.column_stack((np.cos(theta), np.sin(theta))))[0]
    plt.quiver(x[0], y[0], 10*vec_pointing[0], 10*vec_pointing[1], angles='xy', scale_units='xy', scale=1)
    vec_pointing = np.array(np.column_stack((np.cos(theta), np.sin(theta))))[1]
    plt.quiver(x[1], y[1], 10*vec_pointing[0], 10*vec_pointing[1], angles='xy', scale_units='xy', scale=1)
    #plt.plot([50,x[0]],[50,y[0]],'k-')
    
    plt.plot([x[0],x[1]],[y[0],y[1]],'k-')
    

    theta = np.linspace( 0 , 2 * np.pi , 150 )
    radius = 50
    a = 50+ radius * np.cos( theta )
    b = 50+ radius * np.sin( theta )
    plt.plot(a, b, linewidth = 10)

def scatter_period(x,y):
    
    fig = plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
    norm = mpl.colors.Normalize(vmin=0, vmax=2*np.pi)
    #cmap = cm.gist_rainbow
    #m = cm.ScalarMappable(norm=norm, cmap=cmap)
    #temp = m.to_rgba(np.mod(theta,2*np.pi))
    plt.scatter(x,y, s = 1000, alpha = 0.9, marker = 'o',edgecolors='none',cmap = cm.gist_rainbow)
    plt.xlim(0,100)
    plt.ylim(0,100)