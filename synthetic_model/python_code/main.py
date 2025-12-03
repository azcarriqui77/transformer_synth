"""
Generates a synthetic annual database for a transformer with hotspot anomalies 
following a model based on the Stefan-Boltzmann and Newton's cooling laws.
"""

import os
from datetime import datetime
import importlib
import numpy as np
import pandas as pd
from utils import fit_poly5, generate_one_day, anomaly_generation, check_directory

if __name__ == '__main__':
    # Load configuration from external file
    cfg = importlib.import_module('configuration')
    print("Configuration loaded successfully from file: configuration.py")

    # LOADING REAL DATA FOR SYNTHETIC MODEL GENERATION
    
    # Load the already extracted mean temperatures of the regions of interest from a CSV file
    print("Loading already extracted mean temperatures of the regions of interest from a CSV file...")
    week_mean_temps = pd.read_csv(os.path.join(cfg.ruta, 'volumes', 'temperatures', 'CT49014', '2023-10-03.csv'), header=0)
    holiday_mean_temps = pd.read_csv(os.path.join(cfg.ruta, 'volumes', 'temperatures', 'CT49014', '2023-09-30.csv'), header=0)

    week_time_points = list(week_mean_temps['timestamp'])
    holiday_time_points = list(holiday_mean_temps['timestamp'])

    week_mean_temps = week_mean_temps.drop(columns=['timestamp']).to_numpy()
    holiday_mean_temps = holiday_mean_temps.drop(columns=['timestamp']).to_numpy()

    n_clusters = week_mean_temps.shape[1]

    # Load the segmentation mask
    template_df = pd.read_csv(os.path.join(cfg.pathtemplate, 'CT49014/masks.csv'))
    template = np.zeros(shape=cfg.image_size, dtype=int)
    for i, column in enumerate(template_df.columns):
        aux = template_df[column].to_numpy().reshape(cfg.image_size)
        aux = aux * (i + 1)
        template = template + aux
    print(f"Segmentation mask loaded successfully from: {cfg.pathtemplate}")


    # Load the daily ambient temperature data for Almonte (Huelva) for the year 2023
    temp_amb_df = pd.read_csv(os.path.join(cfg.ruta, 'volumes', 'temp_ambience', 'almonte_2023_temp_amb_mod1.csv'))
    # Format the date column
    temp_amb_df['timestamp'] = pd.to_datetime(temp_amb_df['timestamp'].str.slice(0, 19), format="%Y-%m-%d %H:%M:%S")
    # Separate the timestamps column into two columns: date and time
    temp_amb_df['date'] = temp_amb_df['timestamp'].dt.date
    temp_amb_df['time'] = temp_amb_df['timestamp'].dt.time
    # Drop the timestamps column; it is unnecessary
    temp_amb_df.drop(columns=['timestamp'], inplace=True)
    # Drop the row for December 31st, due to lack of data
    temp_amb_df = temp_amb_df[temp_amb_df['date'] != datetime(2023, 12, 31).date()]
    print("Daily ambient temperature data for Almonte for the year 2023 loaded successfully")
    

    # GENERATION OF THE SYNTHETIC ANNUAL TEMPERATURE PROFILE FOR EACH CLUSTER USING THE THERMAL MODEL

    # Create an array to store the generated alarms, and another to store the generated daily temperature series
    alarms = np.zeros((n_clusters - 1, np.unique(temp_amb_df['date'].values).shape[0] , 24*60//5), dtype=bool) # the background (cluster 9) does not generate alarms; 
    alarms[:, :, :] = False # initialize to False
    temp_series_daily = np.zeros((n_clusters, np.unique(temp_amb_df['date'].values).shape[0], 24*60//5), dtype=float) # all regions generate temperature series

    for cluster_id in range(n_clusters): # loop to iterate through each cluster 
        print(f"Generating synthetic annual temperature profile for cluster {cluster_id}...")
        
        # Fitting the 5th-degree polynomial to the daily temperature data and obtaining the parameters of the scaled error normal distribution
        T_pol_1, temp_amb_array1, mu_1, std_1 = fit_poly5(week_mean_temps, week_time_points, temp_amb_df, cluster_id) # weekday
        T_pol_2, temp_amb_array2, mu_2, std_2 = fit_poly5(holiday_mean_temps, holiday_time_points, temp_amb_df, cluster_id) # holiday
        # Save the fitting values to an .npy file
        # check_directory(os.path.join(cfg.pathout, 'fitting_poly5'))
        # np.save(os.path.join(cfg.pathout, 'fitting_poly5', f'poly5_weekday_{cluster_id}.npy'), np.array([T_pol_1]))
        # np.save(os.path.join(cfg.pathout, 'fitting_poly5', f'poly5_holiday_{cluster_id}.npy'), np.array([T_pol_2]))
        print("Correct obtention of the fitting polynomial and parameters of the scaled error normal distribution.")

        # Extrapolation to the full year
        # Assume that the ambient temperature throughout the year is provided in a CSV file and is not constant throughout the day
        print("Generating synthetic temperature profile considering the evolution of the daily ambient temperature")
        for i, day in enumerate(np.unique(temp_amb_df['date'].values)):
            print(f"Generating synthetic daily profile for cluster {cluster_id} on day {day}")
            if day.weekday() < 5: # weekday
                mu = mu_1
                std = std_1
                T_pol = T_pol_1
                Tamb = temp_amb_array1
            else: # holiday
                mu = mu_2
                std = std_2
                T_pol = T_pol_2
                Tamb = temp_amb_array2

            # Get the temperature array for the day to be simulated
            Tambd = temp_amb_df.loc[temp_amb_df['date'] == day, 'T_amb'].to_numpy()
            Tmaxd = np.max(Tambd)
            Tmind = np.min(Tambd)

            # Errors of the polynomial fitting model for that day. Generate a new list of errors
            deltaT = Tmaxd - Tmind
            T_syn = generate_one_day(T_pol, Tamb, Tambd, mu * deltaT, std * deltaT, cfg.termo_params)

            # Hotspot anomaly generation according to the considered thermal model
            if cfg.generate_alarms & (cluster_id != n_clusters - 1): # do not generate anomalies for the background (cluster 9)
                if np.random.rand() < cfg.palarm: # generate an anomaly with probability palarm
                    print(f"Generating hotspot anomaly in cluster {cluster_id} for day {day}...")
                    new_T_syn = anomaly_generation(T_syn, Tambd, cfg.termo_params)
                    alarms_day = new_T_syn != T_syn
                    alarms[cluster_id, i, :] = alarms_day  # mark the positions where anomalies occur
                    print(f"Anomaly generated and alarm registered.")

                else:
                    new_T_syn = T_syn.copy() # do not generate anomalies, the synthetic profile is the same as the one generated by the thermal model
                temp_series_daily[cluster_id, i, :] = new_T_syn
            else:
                temp_series_daily[cluster_id, i, :] = T_syn

    # Save the generated alarms and temperatures to a .csv file for each day of the year
    print("Saving the generated alarms and temperatures to .csv files for each day of the year...")
    for i, day in enumerate(np.unique(temp_amb_df['date'].values)):
        alarms_dict = {}
        temperatures_dict = {}

        # Generate the timestamps for 24 hours every five minutes (288 marks)
        timestamp = temp_amb_df.loc[temp_amb_df['date'] == day, 'time'].astype(str).to_list()
        timestamp = [f"{day} {t}" for t in timestamp]
        alarms_dict['timestamp'] = timestamp
        temperatures_dict['timestamp'] = timestamp

        for cluster_id in range(n_clusters - 1): # the background does not generate alarms
            alarms_dict[cluster_id] = alarms[cluster_id, i, :].astype(int) # convert to int to save to csv
        for cluster_id in range(n_clusters):
            temperatures_dict[cluster_id] = temp_series_daily[cluster_id, i, :]

        alarms_df = pd.DataFrame(alarms_dict)
        temperatures_df = pd.DataFrame(temperatures_dict)
        # Save the generated alarms and temperatures to a .csv file
        check_directory(os.path.join(cfg.pathout, 'alarms'))
        check_directory(os.path.join(cfg.pathout, 'synth_plus_anom_temp'))

        alarms_outfile = os.path.join(cfg.pathout, 'alarms', f'{cfg.prefix}_{day}.csv')
        temperatures_outfile = os.path.join(cfg.pathout, 'synth_plus_anom_temp', f'{cfg.prefix}_{day}.csv')

        alarms_df.to_csv(alarms_outfile, index=False)
        temperatures_df.to_csv(temperatures_outfile, index=False)
        print(f"Saved generated alarms and temperatures for day {day} to .csv files")

    # Generate synthetic images from the generated daily temperature profile and the segmentation mask
    # All pixels in the same cluster region take the value of the calculated temperature for that region at that temporal moment
    # We also save the generated alarms and temperatures to a .csv file
    if cfg.generate_images:
        print("Generating synthetic annual images from the generated daily temperature profile and the segmentation mask...")
        # For each day of the year...
        for i, day in enumerate(np.unique(temp_amb_df['date'].values)):
            print(f"Generating synthetic images for day {day}...")
            check_directory(os.path.join(cfg.pathout, 'synth_plus_anom_images', f"{cfg.prefix}_{day}"))
            day_points = temp_amb_df.loc[temp_amb_df['date'] == day, 'time'].astype(str).to_list()
            # For each temporal moment of the day...
            for j, t in enumerate(day_points):
                time_t = t.replace(':', '-')
                # Initialize the synthetic image
                synthetic_image = np.empty(cfg.image_size, dtype=float)
                synthetic_image[:] = temp_amb_df['T_amb'].to_numpy()[j+i*len(day_points)] + 273.15 # initialize to the ambient temperature at that moment
                # For each cluster...
                for cluster_id in range(n_clusters):
                    temp = temp_series_daily[cluster_id, i, j]
                    # Generate the synthetic image
                    synthetic_image[template == (cluster_id + 1)] = temp
                # Save the generated synthetic image to an .npy file
                out_filename = os.path.join(cfg.pathout, 'synth_plus_anom_images', f"{cfg.prefix}_{day}", f"{cfg.prefix}_{day}_{time_t}.npy")
                np.save(out_filename, synthetic_image)


