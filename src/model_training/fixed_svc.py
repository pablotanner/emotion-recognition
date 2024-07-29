"""
The SVC provided in cuML is broken when both probability=True and class_weight are set.
This is a workaround to fix that issue, an issue and PR have already been opened (and approved) on the cuML repo,
awaiting release.
"""

from cuml.svm import SVC
from cuml.svm.svc import apply_class_weight
from sklearn.calibration import CalibratedClassifierCV
import cuml.internals
from cuml.internals.import_utils import has_sklearn
from cuml.internals.input_utils import input_to_host_array, input_to_cuml_array, input_to_host_array_with_sparse_support

# cuML SVC where probability + class weight combination is fixed
def _fit_proba(self, X, y, sample_weight=None):
    params = self.get_params()
    params["probability"] = False

    # Ensure it always outputs numpy
    params["output_type"] = "numpy"

    # Currently CalibratedClassifierCV expects data on the host, see
    # https://github.com/rapidsai/cuml/issues/2608
    X = input_to_host_array_with_sparse_support(X)
    y = input_to_host_array(y).array

    if not has_sklearn():
        raise RuntimeError(
            "Scikit-learn is needed to use SVM probabilities")


    self.prob_svc = CalibratedClassifierCV(SVC(**params),
                                           cv=5,
                                           method='sigmoid')

    sample_weight = apply_class_weight(self.handle, sample_weight, self.class_weight, y, self.verbose, self.output_type, self.dtype)

    # If sample_weight is not None, it is a cupy array, and we need to convert it to a numpy array for sklearn
    if sample_weight is not None:
        # Currently, fitting a probabilistic SVC with class weights requires at least 3 classes, otherwise the following, ambiguous error is raised:
        # ValueError: Buffer dtype mismatch, expected 'const float' but got 'double'
        if len(set(y)) < 3:
            raise ValueError("At least 3 classes are required to use probabilistic SVC with class weights.")

        # Convert cupy array to numpy array
        sample_weight = sample_weight.get()

    with cuml.internals.exit_internal_api():
        # Fit the model, sample_weight is either None or a numpy array
        self.prob_svc.fit(X, y, sample_weight=sample_weight)
    self._fit_status_ = 0
    return self


SVC._fit_proba = _fit_proba
