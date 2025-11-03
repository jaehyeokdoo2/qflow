from agents.fql import FQLAgent
from agents.ifql import IFQLAgent
from agents.ifql_modified import IFQLModifiedAgent
from agents.iql import IQLAgent
from agents.rebrac import ReBRACAgent
from agents.sac import SACAgent
from agents.fbrac import FBRACAgent
from agents.frebrac import FReBRACAgent
from agents.idm import IDMAgent
from agents.rebrac_idm import ReBRACIDMAgent
from agents.fql_idm import FQLIDMAgent
from agents.fql_separate import FQLSeparateAgent
from agents.lyapunov_fbrac import LyapunovFBRACAgent
from agents.dql import DQLAgent
from agents.idql import IDQLAgent
from agents.fbrac_tdq import FBRAC_TDQAgent
from agents.dql_tdq import DQL_TDQAgent

agents = dict(
    fql=FQLAgent,
    ifql=IFQLAgent,
    ifql_modified=IFQLModifiedAgent,
    iql=IQLAgent,
    fbrac=FBRACAgent,
    rebrac=ReBRACAgent,
    sac=SACAgent,
    frebrac=FReBRACAgent,
    idm=IDMAgent,
    rebrac_idm=ReBRACIDMAgent,
    fql_idm_fixed=FQLIDMAgent,
    fql_separate=FQLSeparateAgent,
    lyapunov_fbrac=LyapunovFBRACAgent,
    dql=DQLAgent,
    idql=IDQLAgent,
    fbrac_tdq=FBRAC_TDQAgent,
    dql_tdq=DQL_TDQAgent,
)
