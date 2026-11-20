"""Same-language estimator orchestration; no R or Stata runtime calls."""
from .assignments import AssignmentSet, assignment_set
from .suite import didsuite, SuiteResult

__version__ = "0.1.0.dev0"
__all__ = ["didsuite", "SuiteResult", "AssignmentSet", "assignment_set"]
