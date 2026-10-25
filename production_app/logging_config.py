"""Central logging configuration."""

import logging

"""Set up root logger once with a consistent format.
    Args:level: Logging level name (e.g., 'INFO', 'DEBUG').
"""


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


"""Return a reusable logger with the given name.
    Args:name: Logger name, typically __name__ of the calling module.
    Returns:A configured logging.Logger instance.
"""


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
