from agents.fql import FQLAgent
from agents.fbrac import FBRACAgent
from agents.qflow import QFlowAgent

agents = dict(
    fbrac=FBRACAgent,
    fql=FQLAgent,
    qflow=QFlowAgent,
)
