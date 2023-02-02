from .emsc import emsc
from .emsc import EMSC, EMSCDetails
from .fringe_emsc import FringeEMSC, FringeEMSCDetails
from .me_emsc import MeEMSC, MeEMSCDetails

try:
    from .dsae import DSAE
except ImportError:
    class DSAE:
        def __getattribute__(self, _):
            raise ImportError(
                'Tensorflow package (>=2.3.4) is required for DSAE')
