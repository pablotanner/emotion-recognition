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