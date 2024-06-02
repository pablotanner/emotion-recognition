
# For the experiment, where I try to find what is more important for score fusion with probability stacking,
# the classifier (train different classifiers with the same feature) or the feature (train the same classifier with different features)
# This code is to visualize it


# Without normal SVC because it can't be used for probability stacking
feature_data = {
    'FACS': {
        #'ProbaSVC': 0.30082937922322794,
        'SVC': 0.4394365847036077,
        'LinearSVC': 0.4189193363161993,
        'RandomForest': 0.40519503669440293,
        'LogisticRegression':0.307269345058387,
        'MLP':  0.4559011428774737,
        'NN': 0.44590382571078063,
        'Stacking': 0.3692547380924225
    },
    'Landmarks': {
        #'ProbaSVC': 0.2913596826796941,
        'SVC': 0.4732473249614243,
        'LinearSVC':  0.4256257484715912,
        'RandomForest':  0.3841663282857978,
        'LogisticRegression': 0.31380542667106126,
        'MLP': 0.44577594287827127,
        'NN':  0.44518033501070975,
        'Stacking': 0.37392884492874146
    },
    'PDM': {
        #'ProbaSVC': 0.3124604609149185,
        'SVC': 0.4764234030931595,
        'LinearSVC': 0.4189193363161993,
        'RandomForest': 0.34047841481769736,
        'LogisticRegression': 0.307269345058387,
        'MLP':  0.45239632107062816,
        'NN': 0.4621321563246723,
        'Stacking':  0.3832770586013794
    },
    'Embedded': {
        #'ProbaSVC': 0.40424141346273734,
        'SVC':  0.4597618607618611,
        'LinearSVC': 0.43354461215148543,
        'RandomForest': 0.3072075959992983,
        'LogisticRegression': 0.31019456956600855,
        'MLP': 0.4518447132536358,
        'NN':  0.45162023655398553,
        'Stacking': 0.4014541804790497
    },
    'HOG': {
        #'ProbaSVC': None,
        'SVC': 0.5099321335489762564,
        'LinearSVC': 0.487737209183920,
        'RandomForest': 0.367283566262456211,
        'LogisticRegression': 0.383532178942321,
        'MLP':  0.5027519378412977,
        'NN': 0.52772918394083,
        'Stacking': 0.438562010387592
    },
}

classifier_data = {
    'LogisticRegression': {
        'HOG': 0.39581261853096295,
        'Landmarks': 0.31459976300727743,
        'PDM': 0.307269345058387,
        'FACS': 0.30226751883669045,
        'Embedded':  0.31019616709176057,
        'Stacking': 0.507920020773825,
    },
    'MLP': {
        'HOG': 0.5027519378412977,
        'Landmarks': 0.44484653558737197,
        'PDM': 0.45239632107062816,
        'FACS':  0.42701160880091726,
        'Embedded': 0.44727511650675755,
        'Stacking':0.5445338873019995,
    },
    'NN': {
        'HOG':  0.5116590738286604,
        'Landmarks':  0.44499356267281487,
        'PDM': 0.4651314841138551,
        'FACS':  0.43297680356478163,
        'Embedded': 0.447288606908663,
        'Stacking':  0.5468709426123085,
    },
    'LinearSVC': {
        'HOG': 0.4934845357399587,
        'Landmarks': 0.421724833226083,
        'PDM': 0.4189193363161993,
        'FACS': 0.4150818239034705,
        'Embedded': 0.43354461215148543,
        'Stacking': 0.5211633341989094,
    },
    'RandomForest': {
        'HOG': 0.3802198084520477,
        'Landmarks':0.38544798751144294,
        'PDM':  0.33737962621494366,
        'FACS': 0.36168514090132753,
        'Embedded':  0.30020760127184215,
        'Stacking':  0.5074006751493119,
    },
    'SVC': {
        'HOG': 0.5099321335489762564,
        'Landmarks': None,
        'PDM': None,
        'FACS': None,
        'Embedded': None,
        'Stacking': 0.5099321335489762564,

    },
}