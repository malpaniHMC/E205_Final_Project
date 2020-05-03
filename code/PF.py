import csv
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import os.path
from scipy import integrate
from scipy.signal import butter,filtfilt, kaiserord, lfilter, firwin, freqz

HEIGHT_THRESHOLD = 0.0  # meters
GROUND_HEIGHT_THRESHOLD = -.4  # meters
DT = 0.01
X_LANDMARK = 5.  # meters
Y_LANDMARK = -5.  # meters
EARTH_RADIUS = 6.3781E6  # meters


def load_data(filename):
    """Load data from the csv log

    Parameters:
    filename (str)  -- the name of the csv log

    Returns:
    data (dict)     -- the logged data with data categories as keys
                       and values list of floats
    """
    is_filtered = False
    if os.path.isfile(filename + "_filtered.csv"):
        f = open(filename + "_filtered.csv")
        is_filtered = True
    else:
        # /Users/computer/Desktop/Spring2020/E205/FinalProject/data/natural gait 1.csv
        pathfinder = "/Users/computer/Desktop/Spring2020/E205/FinalProject/data/"
        f = open(pathfinder+ filename + ".csv")

    file_reader = csv.reader(f, delimiter=',')
    # Load data into dictionary with headers as keys
    data = {}
    header= ["time","gFx","gFy","gFz","ax","ay","az","wx","wy","wz","Azimuth","Pitch","Roll","Latitude","Longitude","Speed (m/s)"]

    for h in header:
        data[h] = []

    row_num = 0
    f_log = open("bad_data_log.txt", "w")
    for row in file_reader:
        for h, element in zip(header, row):
            # If got a bad value just use the previous value
            try:
                data[h].append(float(element))
            except ValueError:
                if(len(data[h])>0):
                    data[h].append(data[h][-1])
                else: 
                    data[h].append(0)
                f_log.write(str(row_num) + "\n")

        row_num += 1
    f.close()
    f_log.close()

    return data, is_filtered


def save_data(data, filename):
    """Save data from dictionary to csv

    Parameters:
    filename (str)  -- the name of the csv log
    data (dict)     -- data to log
    """
    header = ["X", "Y", "Z", "Time Stamp", "Latitude", "Longitude",
              "Yaw", "Pitch", "Roll", "AccelX", "AccelY", "AccelZ"]
    f = open(filename, "w")
    num_rows = len(data["X"])
    for i in range(num_rows):
        for h in header:
            f.write(str(data[h][i]) + ",")

        f.write("\n")

    f.close()


def filter_data(data):
    """Filter lidar points based on height and duplicate time stamp

    Parameters:
    data (dict)             -- unfilterd data

    Returns:
    filtered_data (dict)    -- filtered data
    """

    # Remove data that is not above a height threshold to remove
    # ground measurements and remove data below a certain height
    # to remove outliers like random birds in the Linde Field (fuck you birds)
    filter_idx = [idx for idx, ele in enumerate(data["Z"])
                  if ele > GROUND_HEIGHT_THRESHOLD and ele < HEIGHT_THRESHOLD]

    filtered_data = {}
    for key in data.keys():
        filtered_data[key] = [data[key][i] for i in filter_idx]

    # Remove data that at the same time stamp
    ts = filtered_data["Time Stamp"]
    filter_idx = [idx for idx in range(1, len(ts)) if ts[idx] != ts[idx-1]]
    for key in data.keys():
        filtered_data[key] = [filtered_data[key][i] for i in filter_idx]

    return filtered_data


def convert_gps_to_xy(lat_gps, lon_gps, lat_origin, lon_origin):
    """Convert gps coordinates to cartesian with equirectangular projection

    Parameters:
    lat_gps     (float)    -- latitude coordinate
    lon_gps     (float)    -- longitude coordinate
    lat_origin  (float)    -- latitude coordinate of your chosen origin
    lon_origin  (float)    -- longitude coordinate of your chosen origin

    Returns:
    x_gps (float)          -- the converted x coordinate
    y_gps (float)          -- the converted y coordinate
    """
    x_gps = EARTH_RADIUS*(math.pi/180.)*(lon_gps - lon_origin)*math.cos((math.pi/180.)*lat_origin)
    y_gps = EARTH_RADIUS*(math.pi/180.)*(lat_gps - lat_origin)

    return x_gps, y_gps


def wrap_to_pi(angle):
    """Wrap angle data in radians to [-pi, pi]

    Parameters:
    angle (float)   -- unwrapped angle

    Returns:
    angle (float)   -- wrapped angle
    """
    while angle >= math.pi:
        angle -= 2*math.pi

    while angle <= -math.pi:
        angle += 2*math.pi

    # geq_ind = angle >= math.pi
    return angle

def calc_weight(z_t, particles_t_pred):
    """Calculate the Jacobian of motion model with respect to control input

    Parameters:
    z_t (np.array)     -- lidar measurement
    state (np.array)          -- particle

    Returns:
    w_i (np.array)        -- weight of particle
    """

    """STUDENT CODE START"""
    #need to figure out std deveation of lidar:
    # n = len(particles_t_pred)
    # # print("z", z_t.shape)
    # # print("particles", particles_t_pred.shape)
    # w_i = np.zeros((n,3))
    # sigma_x = np.sqrt(0.268)
    # sigma_y = np.sqrt(0.268)
    # sigma_theta = np.sqrt(0.002)
    # w_i[:,0] = (1/(sigma_x*np.sqrt(2*np.pi)))*np.exp(-((z_t[0]-particles_t_pred[:,0])**2)/(2*sigma_x**2))
    # w_i[:,1] = (1/(sigma_y*np.sqrt(2*np.pi)))*np.exp(-((z_t[1]-particles_t_pred[:,1])**2)/(2*sigma_y**2))
    # w_i[:,2] = (1/(sigma_theta*np.sqrt(2*np.pi)))*np.exp(-(wrap_to_pi(z_t[2]-particles_t_pred[:,2])**2)/(2*sigma_theta**2))
    epsilon= 0.000001 
    w_i = np.zeros(3)
    sigma_x = np.sqrt(0.268)
    sigma_y = np.sqrt(0.268)
    sigma_theta = np.sqrt(0.002)
    w_i[0] = (1/(sigma_x*np.sqrt(2*np.pi)))*np.exp(-((z_t[0]-particles_t_pred[0])**2)/(2*sigma_x**2))
    w_i[1] = (1/(sigma_y*np.sqrt(2*np.pi)))*np.exp(-((z_t[1]-particles_t_pred[1])**2)/(2*sigma_y**2))
    w_i[2] = (1/(sigma_theta*np.sqrt(2*np.pi)))*np.exp(-(wrap_to_pi(z_t[2]-particles_t_pred[2])**2)/(2*sigma_theta**2))
    w = np.prod(w_i)
    if(w<0.000001): 
        w= epsilon
    """STUDENT CODE END"""
    return w


def prediction_step(particles_t_prev, u_t, z_t):
    """Compute the prediction of PF

    Parameters:
    particles_t_prev (np.array)         -- previous belief state
    z_t (np.array)              -- lidar measurement

    Returns:
    particles_t (np.array)          -- the predicted beleif state of time t
    """

    """STUDENT CODE START"""
    n,d = particles_t_prev.shape
    particles_t_state = np.zeros(np.shape(particles_t_prev))
    particles_t_pred = np.zeros((n,d+1))

    perturb = np.random.uniform(-1,1, (n,2)) #MAY WANT TO DECREASE RANDOM RANGE FOR FORWARD DISTANCE
    perturb = np.concatenate((perturb, np.random.uniform(-np.pi, np.pi, (n,1))), axis=1)
    perturb = np.concatenate((perturb, np.zeros((n,2))),axis=1)

    motion_model = np.zeros((n,d))
    motion_model[:,0] = particles_t_prev[:, 3] * DT 
    motion_model[:,1] = particles_t_prev[:, 4] * DT 
    motion_model[:,2] = np.tile([u_t[3]*DT], n)
    motion_model[:,3] = np.tile([u_t[0]*DT], n)
    motion_model[:,4] = np.tile([u_t[1]*DT], n)
    # Progating Motion Model 
    particles_t_state= particles_t_prev  + perturb + motion_model
    particles_t_state[:,2] = np.array([wrap_to_pi(i) for i in particles_t_state[:,2]])

    # Weighting particles
    w_i = np.array([[calc_weight(z_t, particles_t_state[i]) for i in range(n)]]).T
    particles_t_pred= np.concatenate((particles_t_state,w_i), axis = 1)
    """STUDENT CODE END"""

    return particles_t_pred

def resample(particles_t_pred):
    """Calculate predicted measurement based on the predicted state

    Parameters:
    x_bar_t (np.array)  -- the predicted state

    Returns:
    z_bar_t (np.array)  -- the predicted measurement
    """

    """STUDENT CODE START"""
    #NEED TO DO THIS FOR EVERY STATE!!!!
    n,d = particles_t_pred.shape
    weights = list(particles_t_pred[:,d-1])
    weights_sum= np.sum(weights, axis=0)
    weights_sum= [weights_sum for i in range(n)]
    # weight_probs= [1.0/(n)]
    # if(weights_sum==0): 
        # weight_probs = np.tile(weight_probs,n)
    # else:
    weight_probs= list(np.divide(weights, weights_sum))
    choices= np.random.choice(range(0,n), n, p=weight_probs)
    particles_t= particles_t_pred[choices,:]
    """STUDENT CODE END"""
    return particles_t


def correction_step(particles_t_pred):
    """Compute the correction of PF

    Parameters:
    particles_t_pred       (np.array)    -- predicted particles with weights (n,2) (states are a list)

    Returns:
    particles_t       (np.array)    -- the filtered state estimate of time t
    """

    n, d = particles_t_pred.shape
    particles_t = resample(particles_t_pred)
    state_estimate = np.average(particles_t[:,:d], axis=0, weights=particles_t[:,d-1])

    return particles_t[:,:], state_estimate


def main():
    """Run a EKF on logged data from IMU and LiDAR moving in a box formation around a landmark"""

    filepath = ""
    # filename = '../data/position2_10_square1_edited'
    # filename = '../data/position2_front_back_10_edited'
    filename = 'postion2_TJ_1_edited'
    # filename = '../data/position2_TJ_2_edited'
    data, is_filtered = load_data(filepath + filename)

    # Load data into variables
    # header= ["time","gFx","gFy","gFz","ax","ay","az","wx","wy","wz","p","Azimuth","Pitch","Roll","Latitude","Longitude","Speed (m/s)"]
    timestamps = data["time"][3:]
    ax_ddot = data["ax"][3:]
    ax_ddot_var = np.var(ax_ddot[:100])
    ay_ddot = data["ay"][3:]
    ay_ddot_var = np.var(ax_ddot[:100])
    az_ddot = data["az"][3:]
    wx = data["wx"][3:]
    wy = data["wy"][3:]
    wz = data["wz"][3:]
    gFx = data["gFx"][3:]
    gFy = data["gFy"][3:]
    gFz = data["gFz"][3:]
    yaw = data["Azimuth"][3:]
    yaw_init = np.sum(np.array(yaw[500:700]))/200 
    yaw = [wrap_to_pi((-angle -(yaw_init))*math.pi/180) for angle in yaw]
    # yaw = [wrap_to_pi((angle-yaw_init)*math.pi/180) for angle in yaw]
    yaw_var = np.var(yaw[:100])
    lat_gps= data["Latitude"][3:]
    lon_gps= data["Longitude"][3:]


    print("variances")
    print("ax",ax_ddot_var)
    print("ay", ay_ddot_var)
    print("yaw", yaw_var)

    # -----------------xxxxx---------------------------
    sample_rate = 100
    # The Nyquist rate of the signal.
    nyq_rate = sample_rate / 2.0

    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.  We'll design the filter
    # with a 5 Hz transition width.
    width = 5.0/nyq_rate

    # The desired attenuation in the stop band, in dB.
    ripple_db = 60.0

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)

    # The cutoff frequency of the filter.
    cutoff_hz = 0.1

    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))

    # Use lfilter to filter x with the FIR filter.
    ax_ddot = lfilter(taps, 1.0, ax_ddot)
    ay_ddot = lfilter(taps, 1.0, ay_ddot)
    wx = lfilter(taps, 1.0, wx)
    wy = lfilter(taps, 1.0, wy)
    wz = lfilter(taps, 1.0, wz)
    gFx = lfilter(taps, 1.0, gFx)
    gFy = lfilter(taps, 1.0, gFy)
    gFz = lfilter(taps, 1.0, gFz)

    lat_origin = lat_gps[0]
    lon_origin = lon_gps[0]
    X_gps = []
    Y_gps = []

    plt.title("ax")
    plt.plot(yaw)
    plt.plot(ax_ddot)
    plt.show()

    plt.title("ay")
    plt.plot(yaw)
    plt.plot(ay_ddot)
    plt.show()

    plt.title("az")
    plt.plot(yaw)
    plt.plot(az_ddot)
    plt.show()
    
    for i in range(len(lat_gps)):
        x, y = convert_gps_to_xy(lat_gps[i], lon_gps[i], lat_origin, lon_origin)   
        X_gps.append(x)
        Y_gps.append(y)
    X_gps = np.array(X_gps)
    Y_gps = np.array(Y_gps)
    print("X_GPS", np.var(X_gps[:200]))
    print("Y_GPS", np.var(Y_gps[:200]))
    print("origin", lat_origin, lon_origin)
    print("GPS len", len(lat_gps))
    squarex = [0,-10,-10,0,0]
    squarey = [0,0,-10,-10,0]
    # squarey = [0,0,10,10,0]
    # plt.plot(squarex,squarey,label='expected path')
    plt.plot(X_gps, Y_gps, 'o')
    plt.show()

    #  Initialize filter
    """STUDENT CODE START"""
    N = 1000 # number of particles
    initialState = [0,0,0,0,0] # x,y, theta, x_dot, y_dot 
    particles_t_prev_init= np.random.uniform(-5, 15, (N,1)) #initial state assum global (0,0) is at northwest corner
    particles_t_prev_init = np.concatenate((particles_t_prev_init, np.random.uniform(-15,5, (N,1))), axis=1)
    particles_t_prev_init = np.concatenate((particles_t_prev_init, np.random.uniform(-np.pi,np.pi, (N,1))), axis=1)
    zeros = np.zeros((N,2))
    particles_t_prev_init = np.concatenate((particles_t_prev_init, zeros), axis=1)
    particles_t_prev= particles_t_prev_init
    print(particles_t_prev)
    print("particles_prev", particles_t_prev.shape)
    particles = np.zeros((N, len(initialState), len(timestamps)))
    gps_estimates = np.empty((2, len(timestamps)))
    state_estimates = np.zeros((6,len(timestamps)))

    """STUDENT CODE END"""

    #  Run filter over data
    for t, _ in enumerate(timestamps):

        # Get control input
        """STUDENT CODE START"""
        # if(t%500==0):
        #     plt.plot(particles_t_prev[:,0],particles_t_prev[:,1], 'o', label= t)
        #     plt.autoscale()
        #     plt.show()
        z_t = [X_gps[t], Y_gps[t], yaw[t]] 
        transform = np.array([[np.sin(yaw[t]), np.cos(yaw[t]),0,0],[-np.cos(yaw[t]), np.sin(yaw[t]),0,0],[0,0,1,0], [0,0,0,1]])
        u_t = np.array([[ax_ddot[t]],[ay_ddot[t]], [yaw[t]], [wz[t]]])
        u_t = np.dot(transform,u_t) 

        """STUDENT CODE END"""

        # Prediction Step
        particles_t_pred = prediction_step(particles_t_prev, u_t, z_t)
        
        # Correction Step
        particles_t, state_estimate = correction_step(particles_t_pred)

        #Kidnapped Robot 
        # if(np.sum(particles_t[:,3])<1000):
        #     particles_t= particles_t_prev_init
        #  For clarity sake/teaching purposes, we explicitly update t->(t-1)
        particles_t_prev = particles_t[:,:5]

        # Log Data
        particles[:, :, t] = particles_t[:,:5]

        x_gps, y_gps = convert_gps_to_xy(lat_gps=lat_gps[t],
                                         lon_gps=lon_gps[t],
                                         lat_origin=lat_origin,
                                         lon_origin=lon_origin)
        gps_estimates[:, t] = np.array([x_gps, y_gps])
        state_estimate.shape = 6
        #print('u fooked up', state_estimate)
        state_estimates[:,t] = state_estimate
        
        # input('press enter to skirrrrr..')
        plt.ylim(-20,10)
        plt.xlim(-10, 20)
        
        

    """STUDENT CODE START"""
    # Plot or print results here
    plt.plot(state_estimates[0,:],state_estimates[1,:], 'o')
    plt.plot(gps_estimates[0,:], gps_estimates[1,:], 'x')
    plt.autoscale()
    squarex = [0,10,10,0,0]
    squarey = [0,0,-10,-10,0]
    # plt.plot(squarex,squarey,label='expected path')
    plt.show()

    # plt.plot(state_estimates[0,:],state_estimates[1,:],'rx',label='estimates')
    # plt.plot(squarex,squarey,label='expected path')
    # plt.plot(gps_estimates[0,:],gps_estimates[1,:],':',label='GPS Measurements')
    # plt.ylabel('y position (m)')
    # plt.xlabel('x position (m)')
    # plt.legend(loc='best')
    # plt.show()

    #state estimate plot
    # fig, ax = plt.subplots(1,1)
    # ax.plot(state_estimates[0,:],state_estimates[1,:],'r-.',label='estimates')
    # ax.plot(gps_estimates[0,:],gps_estimates[1,:],':',label='GPS Measurements')
    # ax.plot(squarex,squarey,label='expected path')
    # ax.set_xlabel('x position (m)')
    # ax.set_ylabel('y position (m)')
    # ax.legend(loc='best')
    # plt.show()

    #yaw angle over time
    # fig, ax = plt.subplots(1,1)
    # ax.plot(np.arange(len(state_estimates[2,:]))*DT,state_estimates[2,:])
    # ax.set_xlabel('time (s)')
    # ax.set_ylabel('yaw angle (rad)')
    # plt.show()

    #RMS error (not robust)
    # error = []
    # residuals = []
    # for i in range(len(state_estimates[0,:])):
    #     # I tried another way of doing this, but it didn't help
    #     x = state_estimates[0,i]
    #     y = state_estimates[1,i]
    #     min_dist= min([x**2, y**2, (-10-y)**2, (10-x)**2])
    #     # print(min_dist)
    #     distance= min_dist
    #     residuals.append(distance)
    #     error.append(np.sqrt(np.mean(residuals)))
    # # mean = np.mean(residuals)
    # # n= len(residuals)
    # # residuals = [(mean- i)**2/n for i in residuals]
    # # error = np.sqrt(residuals)
    # fig,ax = plt.subplots(1,1)
    # ax.plot(np.arange(len(error))*DT,error)
    # ax.set_xlabel('time (s)')
    # ax.set_ylabel('RMS Tracking Error (m)')
    # plt.show()


    """STUDENT CODE END"""
    return 0


if __name__ == "__main__":
    main()
