import numpy as np
import os
from datetime import datetime
from scipy.stats import norm
from scipy.signal import medfilt
from heat_ode import heat_ode

def check_directory(path: str):
    """
    Checks if the path exists. If it does not exist, it creates it.
    
    Parameters:
        path (str): directory path to check or create.
    
    Returns:
        str: the path of the secured directory.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def get_mean_temperature(image, template, n_clusters):
    """
    Extract mean temperatures for each region of interest (ROI) defined by the template mask.

    Parameters:
    -----------
    image : np.ndarray
        2D array representing the temperature image.
    template : np.ndarray
        2D array representing the segmentation mask with cluster labels.
    n_clusters : int
        Number of clusters (regions of interest).
    Returns:
    --------
    mean_temps : np.ndarray
        1D array of mean temperatures for each cluster. Length is n_clusters.   
    """
    mean_temps = np.zeros(n_clusters)
    for cluster_id in range(n_clusters):
        mask = (template == cluster_id)
        if np.any(mask):
            mean_temps[cluster_id] = np.mean(image[mask])
        else:
            mean_temps[cluster_id] = np.nan  # or some other value indicating no data
    return mean_temps


def get_timeday_from_name(filename):
    """
    Extract the time of day and date from the filename.

    Parameters:
    -----------
    filename : str
        Filename containing the timestamp in the format '..._YYYY-MM-DD_HH-MM-SS.npy'.

    Returns:
    --------
    day_and_time : string
        Day and time.
    """
    basename = os.path.basename(filename)
    parts = basename.split('_')
    if len(parts) >= 3:
        date_part = parts[-2]
        time_part = parts[-1].replace('.npy', '')
        day_and_time = f"{date_part} {time_part.replace('-', ':')}"
        return day_and_time
    else:
        raise ValueError("Filename does not match expected format.")
    
def fit_poly5(temp_array, day_time_points, temp_amb_df, cluster_id):
    """
    Fit a 5th degree polynomial to the given set of temperatures. Also,
    fit the residuals to a normal distribution and return the mean and standard deviation
    considering the scaling of the temperature difference.
    Parameters:
    -----------
    temp_array : np.ndarray
        1D array of temperatures.
    day_time_points : list of str
        List of time points corresponding to the temperatures.
    temp_amb_df: pd.DataFrame
        DataFrame containing ambient temperature data with columns 'dates', 'T_amb', etc.
    cluster_id : int
        Cluster ID for which the polynomial is being fitted.

    Returns:
    --------
    y_fit : np.ndarray
        Fitted polynomial values for the input time points.
    temp_amb_array: np.ndarray
        Array of ambient temperature values for the day when image was taken
    mu_scaled : float
        Mean of the residuals scaled by the temperature difference.
    std_scaled : float
        Standard deviation of the residuals scaled by the temperature difference.   
    """
    # Convert the time points of the day to fractions of the day (0 to 1)
    X_train = []
    for element in day_time_points:
        aux = element.split(' ')[1].split(':')
        # aux = hour/24 + minute/60/24 + second/3600/24
        aux = int(aux[0])/24 + int(aux[1])/60/24 + int(aux[2] if len(aux) > 2 else 0)/3600/24 
        X_train.append(aux)

    y_train = temp_array[0:len(X_train), cluster_id] # just in case the array has more time points. Temperature in Celsius degrees

    # Glitch removal (outliers) in the training data
    # Use a median filter with window 7
    y_train = medfilt(y_train, kernel_size=7)

    # Fit the polynomial
    coeffs = np.polyfit(X_train, y_train, deg=5)
    poly = np.poly1d(coeffs)

    # Generate the fitted values for the day in 5-minute intervals
    y_fit = poly(np.arange(0, 1, 1/(24*60/5))) # Celsius degrees
    
    # Get the approximation errors (e) 
    error = y_train - poly(X_train)   # approximation error. Temperature in Celsius degrees

    # Fit a normal distribution to the obtained error sample
    mu, std = norm.fit(error)
    # Gaussian distribution N((Tmax-Tmin)mu,(Tmax-Tmin)sigma).
    # hypothesis to include the dependence on T amb of the part dependent on the 
    # boundary conditions T amb. Greater temperature increase, greater error distribution 
    # in the model. Selecting T max and T min of the day

    # For the day the images were taken, calculate the difference between T max and T min
    day = day_time_points[0].split(' ')[0] # choose the day and month
    day = datetime.strptime(day, '%Y-%m-%d').date() # format the date to search in the dataframe
    temp_amb_array = (temp_amb_df.loc[temp_amb_df['date'] == day, 'T_amb']).to_numpy() # extract the daily temperatures into an array

    deltaT = np.max(temp_amb_array) - np.min(temp_amb_array) # Celsius degrees

    # Calculate the mean and variance of the errors, removing the temperature difference scaling
    mu_scaled = mu / deltaT
    std_scaled = std / deltaT

    return y_fit, temp_amb_array, mu_scaled, std_scaled

def generate_one_day(T_pol, Tamb, Tambd, mu, std, termo_params):
    """
    Generate a synthetic temperature profile for one day using the thermal model.

    Parameters:
    -----------
    T_pol : np.ndarray
        1D array of polynomial-fitted temperatures for the day.
    Tamb : float or np.ndarray
        Ambient temperatures for the day when images were taken (for model fitting).
    Tambd : float or np.ndarray
        Ambient temperature for the day to be simulated.
    mu : float
        Mean of the residuals scaled by the temperature difference.
    std : float
        Standard deviation of the residuals scaled by the temperature difference.
    termo_params : dict 
        Dictionary containing thermal model parameters.

    Returns:    
    --------
    T_syn : np.ndarray
        1D array of synthetic temperatures for the day. (in Kelvin)
    """
    # Generate the polynomial fitting model errors for that day
    # The errors are independent of the unit of measure (since deltaT ºC = deltaT K)
    errors = np.random.normal(mu, std, size=len(T_pol)) 

    # Modeling the measured T dependent on Tamb for that day
    cte = termo_params['alpha'] * termo_params['emiss'] * termo_params['sigma']
    # Measured radiation emitted by the transformer ROI due to its operation, without considering the ambient energy on the day it was measured
    gamma_transf = cte * ((T_pol + 273.15)**4 - (Tamb + 273.15)**4)

    # Radiation coming from the ambient on that day
    gamma_dia = cte * (Tambd + 273.15)**4

    # Measured temperature of the transformer ROI for the new day to simulate. Add the corresponding measurement error for that day
    T_syn = ((gamma_dia + gamma_transf) / cte) ** (1/4) + errors # Kelvin

    return T_syn # return in Kelvin

def anomaly_generation(T_syn, Tambd, termo_params):
    """
    Generate a temperature anomaly using the thermal ODE model.

    Parameters:
    -----------
    T_syn : float
        Synthetic values of temperature for one day (in Kelvin).
    Tambd : float or np.ndarray
        Ambient temperature for the day to be simulated (in Celsius).
    termo_params : dict 
        Dictionary containing thermal model parameters.

    Returns:    
    --------
    new_T_syn : np.ndarray
        Temperature evolution over time with anomalies for one day (in Kelvin).
    """
    # Find a random moment of the day (avoid the end of the day so the anomaly fits)
    indt = np.random.randint(0, len(T_syn) - 60) # temporal index where the anomaly starts (60 points before * 5 min / point = 300 min = 5 hours)

    # Simulate overheating with the heat_ode function
    # T_syn[indt] is in Kelvin, Tambd is in Celsius
    if isinstance(Tambd, float) or Tambd.size == 1: # if option 1 (constant ambient temp)
        # heatfunction should return something like (ignored, T_anom, ignored)
        time, T_anom, _ = heat_ode(T_syn[indt], Tambd + 273.15, 
                            termo_params['U'], termo_params['m'], termo_params['Cp'], termo_params['A'],
                            termo_params['beta'], termo_params['emiss'], termo_params['sigma'])
    
    else: # if option 2 (varying ambient temp)
        time, T_anom, _ = heat_ode(T_syn[indt], Tambd[indt] + 273.15, # assume ambient temperature is constant during the anomaly (it varies little, it can be assumed)
                            termo_params['U'], termo_params['m'], termo_params['Cp'], termo_params['A'],
                            termo_params['beta'], termo_params['emiss'], termo_params['sigma'])

    # Downsampling to 1 min (decimate)
    decimate = 60 * 1
    T_anom = T_anom[::decimate]  # take one point every minute

    # Overwrite T_syn with anomaly values only in the part of the vector that has values below T_anom
    end_idx = indt + len(T_anom)
    aux = T_syn[indt:end_idx] < T_anom
    new_T_syn = T_syn.copy()
    new_T_syn[indt:end_idx][aux] = T_anom[aux]

    return new_T_syn