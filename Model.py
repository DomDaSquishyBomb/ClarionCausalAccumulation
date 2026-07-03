"""
model.py

Defines the Sequential Causal Reasoning (SCR) agent for the CLARION
architecture, helper functions to run single or multiple trials, and a simple
command-line test harness.
"""

from datetime import timedelta

from pyClarion import (
    Atom,
    Atoms,
    Family,
    Agent,
    Input,
    Choice,
    Pool,
    NumDict,
    Priority,
    Event,
)

from lukasiewicz_rules import LukasiewiczRules
import causal_knowledge  # contains structure‑specific rule sets

# ---------------------------------------------------------------------------#
# Keyspace definition
# ---------------------------------------------------------------------------#


class Event(Atoms):
    """Enumeration of observable events (A, B, C, D)."""

    A: Atom
    B: Atom
    C: Atom
    D: Atom


class CausalModel(Family):
    """Family that groups events for the Luk rule store."""

    event: Event


# ---------------------------------------------------------------------------#
# Agent definition
# ---------------------------------------------------------------------------#


class Participant(Agent):
    """
    SCR agent with a Luk‐logic rule store, working‑memory pool, and choice node.

    Parameters
    ----------
    name : str
        Unique label for this agent instance.

    Notes
    -----
    The agent resolves in two phases:
    1. `start_trial` injects the initial evidence.
    2. `resolve` listens for Luk rule updates and triggers a choice.
    """

    pool: Pool
    p: Family
    d: CausalModel
    luk: LukasiewiczRules
    input: Input
    choice: Choice

    def __init__(self, name: str) -> None:
        p = Family()
        d = CausalModel()
        super().__init__(name, p=p, d=d)

        self.d = d  # re‑expose for typing
        with self:
            self.luk = LukasiewiczRules(f"{name}.luk", d, d, d, d)
            self.input = Input(
                f"{name}.input", self.luk.rules.lhs.chunks, reset=False
            )
            self.pool = Pool(
                f"{name}.pool", p, self.luk.rules.lhs.chunks, func=NumDict.sum
            )
            self.choice = Choice(
                f"{name}.choice", p, self.luk.rules.lhs.chunks
            )

        self.luk.input = self.input.main

        # Connect outputs to the working‑memory pool
        self.pool["luk"] = self.luk.main
        self.pool["input"] = (self.input.main, lambda d: d.scale(x=-1))

        self.choice.input = self.pool.main

    def resolve(self, event: Event) -> None:
        """Trigger a choice whenever Luk rules finish updating."""
        if event.source == self.luk.update:
            self.choice.trigger()

    def start_trial(  # noqa: D401  (imperative verb in summary is okay)
        self,
        dt: timedelta,
        priority: Priority = Priority.PROPAGATION,
    ) -> None:
        """Schedule the start of the next trial."""
        self.system.schedule(self.start_trial, dt=dt, priority=priority)

    def finish_trial(
        self,
        dt: timedelta,
        priority: Priority = Priority.PROPAGATION,
    ) -> None:
        """Schedule the end of the current trial."""
        self.system.schedule(self.finish_trial, dt=dt, priority=priority)


# ---------------------------------------------------------------------------#
# Helper functions
# ---------------------------------------------------------------------------#


def make_participant(
    name: str,
    structure: str = "diamond conjunction",
    sd: float = 1.0,
) -> tuple[Participant, list[Atom]]:
    """
    Initialise an SCR participant pre-loaded with causal knowledge.

    Returns
    -------
    participant : Participant
    events : list[Atom]
        Ordered [A, B, C, D] references for convenience.
    """
    participant = Participant(name)
    events = causal_knowledge.init_knowledge(participant, structure)

    # Set noise level in the choice process
    with participant.choice.params[0].mutable():
        participant.choice.params[0][~participant.choice.p.sd] = sd

    return participant, events


# Handy index map for event names ↔ integer position
_events_index = {"A": 0, "B": 1, "C": 2, "D": 3}


def run_single_trial(
    p_name: str,
    structure: str = "diamond conjunction",
    evidence: str = "A",
    target: str = "D",
    sd: float = 1.0,
) -> dict[str, NumDict]:
    """
    Run one reasoning episode and return trace information.

    Parameters
    ----------
    p_name : str
        Participant identifier.
    structure : str
        Causal structure to load.
    evidence : str
        Initial event injected (e.g., 'A').
    target : str
        Correct terminal event (e.g., 'D').
    sd : float
        Standard deviation for stochastic choice.

    Returns
    -------
    dict
        Keys:
        - reasoning : list[str]  (chosen chunk labels)
        - chosen    : str        (final chunk label)
        - match     : bool       (True if chosen == target)
        - strengths : NumDict    (rule strengths at termination)
    """
    line = [evidence]
    evidence_idx = _events_index[evidence]
    target_idx = _events_index[target]

    p, events = make_participant(p_name, structure, sd)
    evidence_atom = events[evidence_idx]
    target_atom = events[target_idx]
    nil_atom = p.luk.rules.lhs.chunks.nil

    p.start_trial(timedelta())

    while p.system.queue:
        event = p.system.advance()

        if event.source == p.choice.select:
            chosen = next(iter(p.choice.poll().values()))
            line.append(chosen[-1][0])

            if chosen in (~target_atom, ~nil_atom):
                p.finish_trial(timedelta())

            p.input.send({chosen: 1.0})

        if event.source == p.start_trial:
            p.input.send({evidence_atom: 1.0})

        if event.source == p.finish_trial:
            return {
                "reasoning": line,
                "chosen": chosen[-1][0],
                "match": chosen == ~target_atom,
                "strengths": p.luk.strengths[0],
            }

    raise RuntimeError("system.queue emptied without reaching target or nil")


def run_trials(
    n: int = 500,
    structure: str = "diamond conjunction",
    evidence: str = "A",
    target: str = "D",
    sd: float = 1.0,
) -> list[dict[str, NumDict]]:
    """
    Simulate *n* independent participants on the same causal structure.

    Returns
    -------
    list of dict
        One summary dictionary per participant (see `run_single_trial`).
    """
    return [
        run_single_trial(f"p_{i}", structure, evidence, target, sd)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------#
# Quick manual test
# ---------------------------------------------------------------------------#

if __name__ == "__main__":
    outcomes = run_trials(100, "diamond conjunction", evidence="A", target="D", sd=0.5)
    for i, out in enumerate(outcomes, 1):
        print(f"{i}: {out['reasoning']}")