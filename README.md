# Notes
## 22.03.2024
### Evaluation (~4000 Images)
#### Landmarks + Facial Action Units (+200 features)
- SVM: 42.41% Accuracy
- Random Forest: 40.08% Accuracy
#### Fake Features (average)
- SVM/Random Forest: ~10% Accuracy

## 24.03.2024
### Feature Performance
#### All Features
- Highest Accuracy is achieved using all features (double Landmarks, both FAUs) however probably overfititng (around 300 features)
#### FAU Presence (18 features)
- SVM RBF reaches 36.58%
- RF reaches 33.33%
#### FAU Presence + FAU Intensity (35 features)
- SVM Linear 38.78%, RBF 42.28%
- RF 42.67% 


## 25.03.2024
### New Features + Classifiers
#### Landmark Distances
I extracted the landmark distances from the landmark coordinates but this means a feature dimension of +2000 just for the  distances, reached 41.25% with SVM (linear), 38.52% with RF

#### Distances + FAU Presence + FAU Intensity
- SVM RBF 43.06%
- SVM Linear 44.36%
- MLP SGD 42.67%

## 02.04.2024
### New Features: PDM Parameters with rigid and non-rigid face shape
Extracted the PDM parameters using OpenFace, reaching new highest accuracies with only 75 features.
- SVM 45%
- MLP SGD 45.78%

