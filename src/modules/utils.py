import enum
from typing import NamedTuple


class SpikingInputType(enum.Enum):
	spikes = 0
	currents = 1


class SpikingInputSpec(NamedTuple):
	size: int
	iType: SpikingInputType





