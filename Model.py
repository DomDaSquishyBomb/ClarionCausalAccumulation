from pyClarion import Atom, Atoms, Family, Agent, Input, Choice, Pool, Process, ChunkStore, NumDict, Site, Chunk, Priority, Rule, Event
from lukasiewicz_rules import LukasiewiczRules, RuleStore2
from datetime import timedelta

"""Keyspace Definition"""

# These are the events that will be used in the model
class Event(Atoms):
    # Observed events
    A: Atom
    B: Atom
    C: Atom
    D: Atom
    
    
# This is the model that will be used in the agent
class CausalModel(Family):
    event: Event

d = CausalModel()
event = d.event

A = "A" ^ event.A ** event.A
B = "B" ^ event.B ** event.B
C = "C" ^ event.C ** event.C
D = "D" ^ event.D ** event.D

rules = [
        # A -> B
        "B_if_A" ^ 
        A >> B,
        # A -> C
        "C_if_A" ^ 
        A >> C ,
        # B, C -> D
        "D_if_B_AND_C" ^
        (B, C) >> D
    ]


"""Model Construction"""
    
class Participant(Agent):
    pool: Pool
    d: CausalModel
    luk: LukasiewiczRules
    input: Input
    
    def __init__(self, name: str) -> None:
        p = Family()
        
        super().__init__(name, p=p, d=d)
        
        self.d = d
        with self:
            self.luk = LukasiewiczRules(f"{name}.luk", d, d, d, d)
            self.input = Input(f"{name}.input", self.luk.rules.lhs.chunks, reset=False)
            self.pool = Pool(f"{name}.pool", p, self.luk.rules.lhs.chunks, func=NumDict.sum)
            self.choice = Choice(f"{name}.choice", p, self.luk.rules.lhs.chunks)
        
        self.luk.input = self.input.main
        
        
        self.pool["luk"] = (
            self.luk.main
        )
        
        self.pool["input"] = (
            self.input.main,
            lambda d: d.scale(x=-0.5)
        )
    
        self.choice.input = self.pool.main
        
    def resolve(self, event: Event) -> None:
        if event.source == self.luk.update: 
            self.choice.trigger() 
        
    def start_trial(self, 
        dt: timedelta, 
        priority: Priority = Priority.PROPAGATION
    ) -> None:
        self.system.schedule(self.start_trial, dt=dt, priority=priority)
        
    def finish_trial(self, 
        dt: timedelta, 
        priority: Priority = Priority.PROPAGATION
    ) -> None:
        self.system.schedule(self.finish_trial, dt=dt, priority=priority)
        

"""Knowledge Initialization"""

def init_knowledge(participant: Participant) -> None:
    # Learn the rules
    participant.luk.rules.compile(*rules)
    
    
"""Event Processing"""

participant = Participant("participant")
participant.luk.rules.compile(*rules)
results = {
    "condition": None,
    "A": None,
    "B": None,
    "C": None,
    "D": None,
}

evidence = A
target = D
nil = participant.luk.rules.lhs.chunks.nil

participant.start_trial(timedelta())
# We'll send the event A to the working memory and try to retrieve the event D
while participant.system.queue:
    event = participant.system.advance()
    print(event.describe())
    if event.source == participant.choice.select:
        print("WTF is Input Looking like")
        print(participant.input.main[0])
        print("Ok now I know")
        print("WTF is Pool Looking like")
        print(participant.pool.main[0])
        print("Ok now I know")
        picks = participant.choice.poll()
        chosen = next(iter(participant.choice.poll().values()))
        print("Pick:")
        print(chosen)
        if chosen in (~target, ~nil):
            participant.finish_trial(timedelta())
        else:
            participant.input.send({chosen: 1.0})
    if event.source == participant.start_trial:
        participant.input.send({evidence: 1.0})
    if event.source == participant.finish_trial:
        print("Final:")
        print(chosen)
        print("Strengths:")
        print(participant.luk.strengths[0])