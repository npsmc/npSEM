from dataclasses import dataclass
import numpy as np

@dataclass
class TimeSeries:
    time:   np.ndarray
    values: np.ndarray
