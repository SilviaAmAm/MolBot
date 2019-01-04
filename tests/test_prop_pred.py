from sklearn.utils.estimator_checks import check_estimator
from models import properties_pred

def test_sklearn():
    check_estimator(properties_pred.Properties_predictor)

if __name__ == "__main__":
    test_sklearn()