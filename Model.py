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


"""Model Construction"""
    
class Participant(Agent):
    pool: Pool
    d: CausalModel
    luk: LukasiewiczRules
    input: Input
    
    def __init__(self, name: str) -> None:
        p = Family()
        d = CausalModel()
        
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
    event = participant.d.event
    A = "A" ^ event.A ** event.A
    B = "B" ^ event.B ** event.B
    C = "C" ^ event.C ** event.C
    D = "D" ^ event.D ** event.D
    # Learn the rules
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
    participant.luk.rules.compile(*rules)
    return A, B, C, D

def make_participant(name: str, sd: float = 1.0) -> Participant:
    p = Participant(name)              
    events = init_knowledge(p)             
    with p.choice.params[0].mutable():
        p.choice.params[0][~p.choice.p.sd] = sd
    return p, events
    
"""Event Processing"""

events_indx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

def run_single_trial(p_name: str, evidence = 'A', target = 'D', sd: float = 1.0) -> dict[str, NumDict]:
    evidence = events_indx[evidence]
    target = events_indx[target]
    p, event = make_participant(p_name, sd=sd)
    evidence = event[evidence]
    target = event[target]
    nil = p.luk.rules.lhs.chunks.nil
    p.start_trial(timedelta())
    # print(f"{p_name} Starting Pool")
    # print(p.pool.main[0])
    while p.system.queue:
        event = p.system.advance()
        if event.source == p.choice.select:
            chosen = next(iter(p.choice.poll().values()))
            if chosen in (~target, ~nil):
                p.finish_trial(timedelta())
            p.input.send({chosen: 1.0})
        if event.source == p.start_trial:
            p.input.send({evidence: 1.0})
        if event.source == p.finish_trial:
            # print(f"{p_name} End Pool")
            # print(p.pool.main[0])
            # print(f"{p_name} End Choice")
            # print(chosen)
            # print(f"{p_name} End Strengths")
            # print(p.luk.main[0])
            return {
                    "chosen":    chosen,
                    "strengths": p.luk.strengths[0]
                }

    raise RuntimeError("system.queue emptied without reaching target or nil")

def run_trials(n: int = 500,
               *,
               evidence='A',
               target='D',
               sd: float = 1.0) -> list[dict[str, NumDict]]:
    
    return [run_single_trial(f"p_{i}", evidence, target, sd) for i in range(n)]


if __name__ == "__main__":
    outcomes = run_trials(100, sd=1.0)
    for i, out in enumerate(outcomes, 1):
        print(f"{i:>2}: {out['chosen']}")


