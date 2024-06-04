

base_data_path = '/local/scratch/ptanner/concatenated_experiment'

def get_data_path(data_type, feature_name):
    return f'{base_data_path}/{data_type}_{feature_name}.npy'


