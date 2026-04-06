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
    sac=SACAgent,
    iql=IQLAgent,
    rebrac=ReBRACAgent,
    qipo=QIPOAgent,
    fawac=FAWACAgent,
    fbrac=FBRACAgent,
    ifql=IFQLAgent,
    fql=FQLAgent,
    qflow=QFlowAgent,
)
