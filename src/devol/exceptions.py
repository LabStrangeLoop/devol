"""Custom exception hierarchy for the devol package."""


class DevolError(Exception):
    """Base exception for all devol-related errors."""


class ConfigurationError(DevolError):
    """Raised when configuration loading or validation fails."""


class EvolutionError(DevolError):
    """Raised when the evolution process encounters an unexpected failure."""


class FitnessComputationError(DevolError):
    """Raised when a user-provided fitness function fails."""
