class UnsupportedError(ValueError):
    """Requested behavior has no implemented, validated adapter."""


class IdentificationError(ValueError):
    """Requested estimand or inference statistic is undefined."""
