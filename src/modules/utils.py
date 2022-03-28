import collections.abc
import enum
from typing import NamedTuple


class SpikingInputType(enum.Enum):
	spikes = 0
	currents = 1


class SpikingInputSpec(NamedTuple):
	size: int
	iType: SpikingInputType


def mapping_update_recursively(d, u):
	"""
	from https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
	:param d: mapping item that wil be updated
	:param u: mapping item updater
	:return: updated mapping recursively
	"""
	for k, v in u.items():
		if isinstance(v, collections.abc.Mapping):
			d[k] = mapping_update_recursively(d.get(k, {}), v)
		else:
			d[k] = v
	return d
