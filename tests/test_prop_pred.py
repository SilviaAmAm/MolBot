# Copyright (c) Michael Mazanetz (NovaData Solutions LTD.), Silvia Amabilino (NovaData Solutions LTD.,
# University of Bristol), David Glowacki (University of Bristol). All rights reserved.
# Licensed under the GPL. See LICENSE in the project root for license information.

from sklearn.utils.estimator_checks import check_estimator
from models import properties_pred

def test_sklearn():
    check_estimator(properties_pred.Properties_predictor)

if __name__ == "__main__":
    test_sklearn()