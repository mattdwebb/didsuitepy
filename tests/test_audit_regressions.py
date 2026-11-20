import unittest
from didsuitepy.suite import _failure

class EstimatorError(ValueError):
    def __init__(self,reason):
        self.reason=reason

class AuditRegressions(unittest.TestCase):
    def test_structured_rank_and_input_errors_keep_their_meaning(self):
        self.assertEqual(_failure(EstimatorError('numerical_failure')),'NUMERICAL_FAILURE')
        self.assertEqual(_failure(EstimatorError('invalid_input')),'INVALID_INPUT')
        self.assertEqual(_failure(EstimatorError('undefined_estimand')),'IDENTIFICATION_FAILURE')
