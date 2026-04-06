from agents.fql import FQLAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.rebrac import ReBRACAgent
from agents.sac import SACAgent
from agents.fbrac import FBRACAgent
from agents.qflow import QFlowAgent
from agents.fawac import FAWACAgent
from agents.qipo import QIPOAgent

agents = dict(
    fql=FQLAgent,
    ifql=IFQLAgent,
    iql=IQLAgent,
    fbrac=FBRACAgent,
    rebrac=ReBRACAgent,
    sac=SACAgent,
    qflow=QFlowAgent,
    fawac=FAWACAgent,
    qipo=QIPOAgent
)
