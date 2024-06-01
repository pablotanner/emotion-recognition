
# For the experiment, where I try to find what is more important for score fusion with probability stacking,
# the classifier (train different classifiers with the same feature) or the feature (train the same classifier with different features)
# This code is to visualize it


# Without normal SVC because it can't be used for probability stacking
feature_data = {
    'FACS': {
        'ProbaSVC': 0.30082937922322794,
        'LinearSVC': 0.4189193363161993,
        'RandomForest': 0.40519503669440293,
        'LogisticRegression':0.307269345058387,
        'MLP':  0.4559011428774737,
        'NN': 0.44590382571078063,
        'Stacking': 0.3692547380924225
    },
    'Landmarks': {
        'ProbaSVC': 0.2913596826796941,
        'LinearSVC': 0.4217226720638964,
        'RandomForest': 0.38804498586189706,
        'LogisticRegression': 0.3148474340711257,
        'MLP': 0.3148474340711257,
        'NN': 0.44205240616177544,
        'Stacking': 0.3625032603740692
    },
    'PDM': {
        'ProbaSVC': 0.3124604609149185,
        'LinearSVC':  0.4189193363161993,
        'RandomForest': 0.337906768543917,
        'LogisticRegression': 0.307269345058387,
        'MLP': 0.44854294908748243,
        'NN': 0.46403045512672647,
        'Stacking':  0.3804206848144531
    },
    'Embedded': {
        'ProbaSVC': 0.40424141346273734,
        'LinearSVC':0.43354461215148543,
        'RandomForest': 0.30798132888716295,
        'LogisticRegression':0.31019616709176057,
        'MLP': 0.45156378433595057,
        'NN':  0.45162023655398553,
        'Stacking': 0.4014541804790497
    },
    'HOG': {
        'ProbaSVC': None,
        'LinearSVC': None,
        'RandomForest': None,
        'LogisticRegression': None,
        'MLP': None,
        'NN': None,
        'Stacking': None
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
    },
    'ProbaSVC': {

    },
}