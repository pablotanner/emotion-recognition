import numpy as np


# Expects features to be extracted from the image

def predict_emotions(feature_dict, spatial_pipeline, facs_pipeline, pdm_pipeline, hog_pipeline, stacking_model):
    # Extract features from the image feature dictionary
    spatial_features = np.concatenate([feature_dict['landmarks'], feature_dict['landmarks_3d']], axis=1)
    facs_features = np.concatenate([feature_dict['facs_intensity'], feature_dict['facs_presence']], axis=1)
    pdm_features = np.array(feature_dict['nonrigid_face_shape'])
    hog_features = np.array(feature_dict['hog'])

    # Predict probabilities using base models
    spatial_probabilities = spatial_pipeline.predict_proba(spatial_features)
    facs_probabilities = facs_pipeline.predict_proba(facs_features)
    pdm_probabilities = pdm_pipeline.predict_proba(pdm_features)
    hog_probabilities = hog_pipeline.predict_proba(hog_features)

    # Stack probabilities
    stacked_features = np.column_stack(
        (spatial_probabilities, facs_probabilities, pdm_probabilities, hog_probabilities))

    # Predict final emotions using the stacking model
    final_prediction = stacking_model.predict(stacked_features)

    return final_prediction

