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

## 19.04.2024
### Standardize 3D Landmarks
- Standardized 3D landmarks in terms of rotation and positioned, reached 46.17% with SVM Linear in combination with FAU intensity and nonrigid-shape (NO STANDARDSCALER)

## 25.04.2024
### Ran tests using Embeddings extracted from various Face Recognition Models
| Model           | Accuracy (LFW) | EmoRec Accuracy (AffectNet) | Embedding Size |
|-----------------|----------------|-----------------------------|----------------|
| VGG-Face        | 98.9%          | 34.88%                      | 4096           |
| Facenet         | 99.2%          | 32.00%                      | 128            |
| Facenet512      | 99.6%          | 26.38%                      | 512            |
| OpenFace        | 92.9%          | 26.12%                      | 129            |
| DeepFace (FB)   | 97.35%         | 32.38%                      | 4096           |
| DeepID          | 97.4%          | 20.75%                      | 160            |
| ArcFace         | 99.5%          | 32.00%                      | 512            |
| SFace           | 99.5%          | 32.75%                      | 128            |
Seems to be a positive correlation between how a model performs in FaceRec and how well its extracted embeddings perform in EmoRec 