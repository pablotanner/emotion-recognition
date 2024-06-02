from cuml.svm import SVC
from cuml.svm.svc import apply_class_weight
from sklearn.calibration import CalibratedClassifierCV
import cuml.internals
from cuml.internals.import_utils import has_sklearn
from cuml.internals.input_utils import input_to_host_array, input_to_cuml_array, input_to_host_array_with_sparse_support

# cuML SVC where probability + class weight combination is fixed

def _fit_proba(self, X, y, sample_weight, convert_dtype=True):
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

    convert_to_dtype = self.dtype if convert_dtype else None
    y_m, _, _, _ = \
        input_to_cuml_array(y, check_dtype=self.dtype,
                            convert_to_dtype=convert_to_dtype,
                            check_rows=self.n_rows, check_cols=1)

    sample_weight = apply_class_weight(self.handle, sample_weight, self.class_weight, y_m, self.verbose, self.output_type, self.dtype)

    with cuml.internals.exit_internal_api():
        self.prob_svc.fit(X, y, sample_weight=sample_weight)
    self._fit_status_ = 0
    return self

