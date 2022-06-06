from .emsc import emsc  # noqa: F401
from .emsc import EMSC, EMSCInternals  # noqa: F401
from .fringe_emsc import FringeEMSC, FringeEMSCInternals  # noqa: F401
from .me_emsc import MeEMSC, MeEMSCInternals  # noqa: F401

try:
    from .dsae import DSAE
except ImportError:
    class DSAE:
        def __getattribute__(self, _):
            raise ImportError(
                'Tensorflow package (>=2.3.4) is required for DSAE')
