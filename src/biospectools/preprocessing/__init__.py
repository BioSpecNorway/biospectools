from .emsc import emsc
from .emsc import EMSC, EMSCInternals
from .fringe_emsc import FringeEMSC, FringeEMSCInternals
from .me_emsc import MeEMSC, MeEMSCInternals

try:
    from .dsae import DSAE
except ImportError:
    class DSAE:
        def __getattribute__(self, _):
            raise ImportError(
                'Tensorflow package (>=2.3.4) is required for DSAE')
