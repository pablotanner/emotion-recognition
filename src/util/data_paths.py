

base_data_path = '/local/scratch/ptanner/concatenated_experiment'

def get_data_path(data_type, feature_name):
    if feature_name == 'hog':
        feature_name = 'hog2'
    return f'{base_data_path}/{data_type}_{feature_name}.npy'


