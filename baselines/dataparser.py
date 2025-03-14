import os
import json
from collections import defaultdict
from datetime import datetime
import numpy as np


def extract_date_from_filename(filename) -> datetime:
    # Extract the date part from the filename
    try:
            # Assuming the filename structure is scena_x-YYYYMMDD-HHMMSS.json
        date_str = filename.split('-')[1]  # Extract YYYYMMDD
        time_str = filename.split('-')[2]  # Extract HHMMSS.json
        datetime_str = f"{date_str}-{time_str.split('.')[0]}"
        return datetime.strptime(datetime_str, "%Y%m%d-%H%M%S")
    except IndexError:
        # Handle the case where the filename structure doesn't match expectations
        return None
    

class Measurement:
    def __init__(self, json_file_path):
        self.accel: np.ndarray = None
        self.gyro : np.ndarray = None
        self.oxy  : np.ndarray = None
        self.date : datetime = extract_date_from_filename(json_file_path)
        self.filename = json_file_path
        self._load_json(json_file_path)
    
    def _load_json(self, json_file_path):
        # Load JSON data
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        # Extract the first "video istruzioni ..." key
        measurements = data.get("measurements", {})
        for key, value in measurements.items():
            if key.startswith("video_istruzioni"):
                # Extract "accel data", "gyro_data", and "oxy_data"
                accel = value.get("accel_data", {})
                self.accel = self._extract_time_series(accel)
                
                # Optionally extract other data if needed
                gyro = value.get("gyro_data", {})
                self.gyro = self._extract_time_series(gyro)
                
                oxy = value.get("oxy_data", {})
                self.oxy = self._extract_time_series(oxy)
                
                break
   
    
    def _extract_time_series(self, data_dict):
        # Convert time series data into a sorted list of tuples
        time_series = sorted((int(timestamp), values) for timestamp, values in data_dict.items())
        # Extract only the values and convert them into a numpy array
        data_array = np.array([values for _, values in time_series])
        return data_array
    
    
    def __str__(self):
        return f"********************************************\nMeasurement date: {self.date}\nFilename: {self.filename})\naccel shape: {self.accel.shape}\ngyro shape: {self.gyro.shape}\noxy shape: {self.oxy.shape}"
    
    def __repr__(self):
        return self.__str__()


def read_database(base_path):
    data = defaultdict(lambda: defaultdict(list))

    # Iterate over object directories
    for object_id in os.listdir(base_path):
        object_path = os.path.join(base_path, object_id)
        if os.path.isdir(object_path):
            # Iterate over scene directories
            for scene_dir in os.listdir(object_path):
                scene_path = os.path.join(object_path, scene_dir)
                if os.path.isdir(scene_path):
                    # Identify scene key (s1, s2, etc.)
                    scene_key = scene_dir.split('_')[1] if '_' in scene_dir else scene_dir
                    scene_key = int(scene_key)  # Extract the number from the key
                    
                    # Collect JSON files and sort them by date
                    json_files = []
                    for filename in os.listdir(scene_path):
                        if filename.endswith('.json'):
                            date = extract_date_from_filename(filename)
                            if date:
                                json_files.append((date, filename))
                      
                    
                    # Sort JSON files by date
                    json_files.sort(key = lambda x: x[0], reverse=True)

                    for _, json_filename in json_files:
                        json_file_path = os.path.join(scene_path, json_filename)
                        try:
                            measurement = Measurement(json_file_path)
                            data[object_id][scene_key].append(measurement)
                        except Exception as e:
                            print(f"Error processing {json_file_path}: {e}")
                    
                    
                    # Add sorted filenames to the data structure, just appending fileneme string
                    #data[object_id][scene_key].extend([file[1] for file in json_files])
    
    return data

def filter_dataset(dataset, verbose=1):
    """
    Filter the dataset to take the first measurement for each scene.
    If the key does not have all 5 scenes, it will be removed from the dataset.
    """
    filtered = {}
    error_keys = []
    for db_id in dataset.keys():
        filtered[db_id] = {}
        try:
            for table_id in range(1, 6):
                filtered[db_id][table_id] = dataset[db_id][table_id][0]
        except IndexError:
            if verbose > 0:
                print("IndexError on key: ", db_id, "Scene: ", table_id)
            error_keys.append(db_id)

    # remove the keys that caused the error
    for key in error_keys:
        del filtered[key]
    
    return filtered

def compact_data(data, gyro_only=False):
    compacted = []
    labels = []
    for db_id in data.keys():
        for scene_id in data[db_id].keys():
            if not gyro_only:
                
                accel = data[db_id][scene_id].accel[200:1450]
                gyro = data[db_id][scene_id].gyro[200:1450]
                compacted.append(np.concatenate([accel, gyro], axis=1))
            else:
                gyro = data[db_id][scene_id].gyro[200:1450]
                compacted.append(gyro)
        
            if db_id.startswith('1'):
                labels.append(1)
            else:
                labels.append(0)

    return np.array(compacted), np.array(labels)


