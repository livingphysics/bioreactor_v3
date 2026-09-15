"""Load hardware/config only when requested; pure control modules work off-Pi."""
from importlib import import_module


def __getattr__(name):
    if name == 'Bioreactor':
        return import_module('.bioreactor', __name__).Bioreactor
    if name == 'Config':
        return import_module('.config', __name__).Config
    if name == 'utils':
        return import_module('.utils', __name__)
    raise AttributeError(name)

__all__ = ['Bioreactor', 'Config', 'utils']
