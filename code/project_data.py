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
DT = 0.1
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
        f = open(filename + ".csv")

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
    return angle

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
    # x_gps = EARTH_RADIUS*(math.pi/180.)*(lon_gps - lon_origin)*math.cos((math.pi/180.)*lat_origin)
    # y_gps = EARTH_RADIUS*(math.pi/180.)*(lat_gps - lat_origin)
    x_gps = np.abs(EARTH_RADIUS*(math.pi/180.)*(lon_gps - lon_origin)*math.cos((math.pi/180.)*lat_origin))
    y_gps = np.abs(EARTH_RADIUS*(math.pi/180.)*(lat_gps - lat_origin))
    return x_gps, y_gps

def propogate_state(x_t_prev, u_t):
    """Propogate/predict the state based on chosen motion model

    Parameters:
    x_t_prev (np.array)  -- the previous state estimate
    u_t (np.array)       -- the current control input

    Returns:
    x_bar_t (np.array)   -- the predicted state
    """
    #nonlinear motion model from whiteboard- in terms of u and xt-1
    (n_x,)= x_t_prev.shape
    x_bar_t = np.zeros(n_x)
    x_bar_t[0] = x_t_prev[0]+x_t_prev[3]*DT
    x_bar_t[1] = x_t_prev[1]+x_t_prev[4]*DT
    x_bar_t[2] = wrap_to_pi(x_t_prev[2]+(x_t_prev[5]*DT))
    x_bar_t[3] = x_t_prev[3]+u_t[0]*DT
    x_bar_t[4] = x_t_prev[4]+u_t[1]*DT
    x_bar_t[5] = wrap_to_pi(u_t[2]-x_t_prev[2])/DT #look into if correctly used
    x_bar_t[6] = x_t_prev[2]
    # x_bar_t[6] = u_t[2]
    return x_bar_t


def calc_prop_jacobian_x(x_t_prev, u_t):
    """Calculate the Jacobian of your motion model with respect to state

    Parameters:
    x_t_prev (np.array) -- the previous state estimate
    u_t (np.array)      -- the current control input

    Returns:
    G_x_t (np.array)    -- Jacobian of motion model wrt to x
    """
    (n_x,) = x_t_prev.shape
    (n_u, d_u) = u_t.shape
    G_x_t = np.empty((n_x, n_x))  # add shape of matrix
    G_x_t= [[1,0,0,DT,0,0,0],
            [0,1,0,0,DT,0,0],
            [0,0,1,0,0,DT,0],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,0,(-1.0/DT)],
            [0,0,1,0,0,0,0],]

    return G_x_t


def calc_prop_jacobian_u(x_t_prev, u_t):
    """Calculate the Jacobian of motion model with respect to control input

    Parameters:
    x_t_prev (np.array)     -- the previous state estimate
    u_t (np.array)          -- the current control input

    Returns:
    G_u_t (np.array)        -- Jacobian of motion model wrt to u
    """

    """STUDENT CODE START"""
    (n_x,)= x_t_prev.shape
    n_u, d_u= u_t.shape
    G_u_t = np.zeros((n_x,n_u)) # add shape of matrix
    G_u_t[0] = [0, 0, 0]
    G_u_t[1] = [0, 0, 0]
    G_u_t[2] = [0, 0, 0]
    G_u_t[3] = [DT,0, 0]
    G_u_t[4] = [0,DT, 0]
    G_u_t[5] = [0,0,(1.0/DT)]
    G_u_t[6] = [0,0, 0]
    """STUDENT CODE END"""

    return G_u_t


def prediction_step(x_t_prev, u_t, sigma_x_t_prev):
    """Compute the prediction of EKF

    Parameters:
    x_t_prev (np.array)         -- the previous state estimate
    u_t (np.array)              -- the control input
    sigma_x_t_prev (np.array)   -- the previous variance estimate

    Returns:
    x_bar_t (np.array)          -- the predicted state estimate of time t
    sigma_x_bar_t (np.array)    -- the predicted variance estimate of time t
    """

    """STUDENT CODE START"""
    # Covariance matrix of control input
    sigma_u_t = np.array([[.8763,0,0],[0,.59221,0],[0,0,.002]])  # write sigma_u_t !!!!
    #sigma_u_t = np.array([[1000,0,0],[0,1000,0],[0,0,1000]])
    #transform = np.array([[np.cos(x_t_prev[2]), -np.sin(x_t_prev[2])],[np.sin(x_t_prev[2]), np.cos(x_t_prev[2])]])
    #sigma_u_t = np.matmul(transform,sigma_u_t_local)
    #sigma_u_t = np.array([[.8+np.absolute(x_t_prev[5]),0],[0, 0.6 +np.absolute(x_t_prev[5])]])
    G_x_t = calc_prop_jacobian_x(x_t_prev,u_t)
    G_u_t = calc_prop_jacobian_u(x_t_prev, u_t)
    x_bar_t = propogate_state(x_t_prev, u_t)
    sigma_x_bar_t = np.matmul(np.matmul(G_x_t,sigma_x_t_prev),np.transpose(G_x_t))
    sigma_x_bar_t = np.add(sigma_x_bar_t,np.matmul(np.matmul(G_u_t,sigma_u_t),np.transpose(G_u_t)))
    """STUDENT CODE END"""

    return [x_bar_t, sigma_x_bar_t]


def calc_meas_jacobian(x_bar_t):
    """Calculate the Jacobian of your measurment model with respect to state

    Parameters:
    x_bar_t (np.array)  -- the predicted state

    Returns:
    H_t (np.array)      -- Jacobian of measurment model
    """
    (n_x,)= x_bar_t.shape
    H_t = np.zeros((3, n_x))
    H_t[0] = [1,0,0,0,0,0,0]
    H_t[1] =[0,1,0,0,0,0,0]
    H_t[2] = [0,0,1,0,0,0,0]
    return H_t


def calc_kalman_gain(sigma_x_bar_t, H_t):
    """Calculate the Kalman Gain

    Parameters:
    sigma_x_bar_t (np.array)  -- the predicted state covariance matrix
    H_t (np.array)            -- the measurement Jacobian

    Returns:
    K_t (np.array)            -- Kalman Gain
    """
    """STUDENT CODE START"""
    # Covariance matrix of measurments
    sigma_z_x= np.array([0.01,0,0]) #CALC SIGMA ZZZ
    sigma_z_y= np.array([0, 0.01, 0])
    sigma_z_theta = np.array([0, 0, 0.0002])
    sigma_z_t = np.array([sigma_z_x,sigma_z_y,sigma_z_theta])
    sigmax_dot_h = np.matmul(sigma_x_bar_t,H_t.T)
    inverse = np.linalg.inv(np.add(np.matmul(np.matmul(H_t,sigma_x_bar_t),H_t.T),sigma_z_t))
    K_t = np.matmul(sigmax_dot_h,inverse)
    """STUDENT CODE END"""

    return K_t


def calc_meas_prediction(x_bar_t):
    """Calculate predicted measurement based on the predicted state

    Parameters:
    x_bar_t (np.array)  -- the predicted state

    Returns:
    z_bar_t (np.array)  -- the predicted measurement
    """

    # (n,d) = x_bar_t.shape
    z_bar_t = np.zeros((3,1))

    z_bar_t[0] = [x_bar_t[0]]
    z_bar_t[1] = [x_bar_t[1]]
    z_bar_t[2] = [x_bar_t[2]]
    
    return z_bar_t

def correction_step(x_bar_t, z_t, sigma_x_bar_t):
    """Compute the correction of EKF

    Parameters:
    x_bar_t       (np.array)    -- the predicted state estimate of time t
    z_t           (np.array)    -- the measured state of time t
    sigma_x_bar_t (np.array)    -- the predicted variance of time t

    Returns:
    x_est_t       (np.array)    -- the filtered state estimate of time t
    sigma_x_est_t (np.array)    -- the filtered variance estimate of time t
    """

    (n_x,) = x_bar_t.shape
    H_t = calc_meas_jacobian(x_bar_t)
    K = calc_kalman_gain(sigma_x_bar_t,H_t)
    measDiff = np.subtract(z_t,calc_meas_prediction(x_bar_t))
    measDiff[2] = wrap_to_pi(measDiff[2])
    x_bar_t.shape = (n_x,1)
    x_est_t = np.add(x_bar_t,np.matmul(K,measDiff))
    x_est_t[2] = wrap_to_pi(x_est_t[2])
    sigma_x_est_t = np.matmul(np.subtract(np.eye(n_x,n_x),np.matmul(K,H_t)),sigma_x_bar_t)

    return [x_est_t, sigma_x_est_t]

def main():
    """Run a EKF on logged data from IMU and LiDAR moving in a box formation around a landmark"""

    filepath = ""
    filename = '../data/position1_10_square_edited'
    # filename = '../data/position1_front_back_10_edited'
    data, is_filtered = load_data(filepath + filename)

    # Load data into variables
    # header= ["time","gFx","gFy","gFz","ax","ay","az","wx","wy","wz","p","Azimuth","Pitch","Roll","Latitude","Longitude","Speed (m/s)"]
    timestamps = data["time"][3:]
    ax_ddot = data["ax"][3:]
    ay_ddot = data["ay"][3:]
    az_ddot = data["az"][3:]
    wx = data["wx"][3:]
    wy = data["wy"][3:]
    wz = data["wz"][3:]
    gFx = data["gFx"][3:]
    gFy = data["gFy"][3:]
    gFz = data["gFz"][3:]
    yaw = data["Azimuth"]
    yaw_init = np.sum(np.array(yaw[:100]))/100
    yaw = [wrap_to_pi((angle+(360-yaw_init))*math.pi/180) for angle in yaw[3:]]
    lat_gps= data["Latitude"][3:]
    lon_gps= data["Longitude"][3:]
    plt.plot(ax_ddot)
    plt.show() 
    plt.plot(ay_ddot)
    plt.show()
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
    # ----------------xxxxxx 

    # plt.title("Yaw")
    # plt.plot(yaw)
    # plt.show()

    plt.title("x_ddot")
    plt.plot(ax_ddot)
    plt.show()

    # plt.title("x_ddot filtered")
    # plt.plot(filtered_x)
    # plt.show()

    # plt.title("y_ddot")
    # plt.plot(ay_ddot)
    # plt.show()

    # plt.title("z_ddot")
    # plt.plot(az_ddot)
    # plt.show()

    # plt.plot(yaw)
    # plt.title("wx")
    # plt.plot(wx)
    # plt.show()

    # plt.plot(yaw)
    # plt.title("wy")
    # plt.plot(wy)
    # plt.show()

    plt.plot(yaw)
    plt.title("wz")
    plt.plot(wz)
    plt.show()

    lat_origin = lat_gps[0]
    lon_origin = lon_gps[0]
    X_gps = []
    Y_gps = []
    # plot gps: 
    
    for i in range(len(lat_gps)):
        x, y = convert_gps_to_xy(lat_gps[i], lon_gps[i], lat_origin, lon_origin)   
        X_gps.append(x)
        Y_gps.append(y)
    X_gps = np.array(X_gps)
    Y_gps = np.array(Y_gps)
    print("origin", lat_origin, lon_origin)
    print("GPS len", len(lat_gps))
    squarex = [0,10,10,0,0]
    squarey = [0,0,10,10,0]
    plt.plot(squarex,squarey,label='expected path')
    plt.plot(X_gps, Y_gps, 'o')
    plt.show()

    #  Initialize filter
    N = 7 # number of states
    state_est_t_prev = np.array([0,0,0,0,0,0,0]) #initial state assum global (0,0) is at northwest corner
    var_est_t_prev = np.identity(N)

    state_estimates = np.zeros((N, len(timestamps)))
    covariance_estimates = np.zeros((N, N, len(timestamps)))
    gps_estimates = np.empty((2, len(timestamps)))

    state_estimates[:,-1] = state_est_t_prev
    covariance_estimates[:,:,-1] = var_est_t_prev

    #  Run filter over data
    for t, _ in enumerate(timestamps):
        if(t!=0): 
            DT = timestamps[t]-timestamps[t-1]
            if(DT==0):
                continue
        
        # Get control input
        transform = np.array([[np.sin(state_est_t_prev[2]), np.cos(state_est_t_prev[2]),0],[-np.cos(state_est_t_prev[2]), np.sin(state_est_t_prev[2]),0],[0,0,1]])
        u_t = np.array([[0], [0],[yaw[t]]])
        # if(wz[t]>0.5):
        #     u_t = np.array([[-state_est_t_prev[3]], [-state_est_t_prev[4]], [yaw[t]]])
        u_t_global = np.matmul(transform,u_t)
 
        state_est_t_prev = state_estimates[:,t-1]
        var_est_t_prev = covariance_estimates[:,:,t-1]

        # Prediction Step
        state_pred_t, var_pred_t = prediction_step(state_est_t_prev, u_t_global, var_est_t_prev)

        # Get measurement
        z_t = np.array([[X_gps[t]], [Y_gps[t]], [yaw[t]]])

        #Correction Step
        state_est_t, var_est_t = correction_step(state_pred_t, z_t, var_pred_t)
        # if(t>=1700 and t<2000):
            # plt.arrow(state_est_t[0][0], state_est_t[1][0], np.cos(state_est_t[2][0]), np.sin(state_est_t[2][0]))
        #  For clarity sake/teaching purposes, we explicitly update t->(t-1)
        state_est_t_prev = state_est_t
        var_est_t_prev = var_est_t

        # Log Data
        state_est_t.shape = (7,)
        state_estimates[:, t] = state_est_t
        covariance_estimates[:, :, t] = var_est_t

        x_gps, y_gps = convert_gps_to_xy(lat_gps=lat_gps[t],
                                         lon_gps=lon_gps[t],
                                         lat_origin=lat_origin,
                                         lon_origin=lon_origin)
        gps_estimates[:, t] = np.array([x_gps, y_gps])
        # if(t%100==0):
        #     plt.plot(squarex,squarey,label='expected path')
        #     plt.plot(state_est_t[0], state_est_t[1], 'o')
        #     plt.show()

    plt.title("x_dot")
    plt.plot(yaw)
    plt.plot(state_estimates[3,:])
    plt.show()
    plt.title("y_dot")
    plt.plot(yaw)
    plt.plot(state_estimates[4,:])
    plt.show()

    plt.plot(state_estimates[0,:],state_estimates[1,:],'o',label='estimates')
    # plt.arrow(state_estimates[0,:50],state_estimates[1,:50], cos_theta[:50], sin_theta[:50])
    plt.plot(squarex,squarey,label='expected path')
    # plt.plot(gps_estimates[0,:],gps_estimates[1,:],':',label='GPS Measurements')
    plt.ylabel('y position (m)')
    plt.xlabel('x position (m)')
    plt.legend(loc='best')
    plt.show()

main()
