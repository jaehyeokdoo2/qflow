from agents.fql import FQLAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.rebrac import ReBRACAgent
from agents.sac import SACAgent
from agents.fbrac import FBRACAgent
from agents.dql import DQLAgent
from agents.idql import IDQLAgent
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
    dql=DQLAgent,
    idql=IDQLAgent,
    qflow=QFlowAgent,
    qipo=QIPOAgent
)
