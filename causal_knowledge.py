"""
knowledge_init.py

Helper functions that inject structure-specific causal rules into a
`Participant` agent.  Each function returns a tuple of chunk references
(in the order A, B, …) so calling code can access them directly.
"""

from typing import List

from pyClarion import Chunk

from model import Participant

# --------------------------------------------------------------------------- #
# Public interface
# --------------------------------------------------------------------------- #


def init_knowledge(participant: Participant, structure: str) -> List[Chunk]:
    """
    Load causal rules for *structure* into *participant* and return event chunks.

    Parameters
    ----------
    participant : Participant
        The agent to which rules are added.
    structure : {"fork", "diamond conjunction", "diamond disjunction",
                 "chain three", "chain four", "imply"}
        Name of the causal structure.

    Returns
    -------
    list[Chunk]
        Event chunks (A, B, C, D as available) for later reference.

    Raises
    ------
    ValueError
        If *structure* is not one of the recognised names.
    """
    mapping = {
        "fork": init_knowledge_fork,
        "diamond conjunction": init_knowledge_diamond_conjunction,
        "diamond disjunction": init_knowledge_diamond_disjunction,
        "chain three": init_knowledge_chain_three,
        "chain four": init_knowledge_chain_four,
        "imply": init_knowledge_imply,
        "spoon disjunction": init_knowledge_spoon_disjunction,
        "ladder": init_knowledge_ladder,
        "ladder plus": init_knowledge_ladder_plus,
        "net": init_knowledge_net,
        "tall glass": init_knowledge_tall_glass,
        "two pyramids": init_knowledge_two_pyramids,
        "highway entrance": init_knowledge_highway_entrance,
        "big worm small worm": init_knowledge_bigworm_small_worm,
        "crossroads": init_knowledge_crossroads,
    }

    try:
        return mapping[structure](participant)
    except KeyError as exc:
        raise ValueError(f"Unknown structure: {structure}") from exc


# --------------------------------------------------------------------------- #
# Individual structure loaders
# --------------------------------------------------------------------------- #


def init_knowledge_imply(participant: Participant) -> List[Chunk]:
    """A → B."""
    event = participant.d.event
    A = "A" ^ event.A ** event.A
    B = "B" ^ event.B ** event.B

    rules = [
        "B_if_A" ^ A >> B,  # A → B
    ]
    participant.luk.rules.compile(*rules)
    return [A, B]


def init_knowledge_fork(participant: Participant) -> List[Chunk]:
    """A branches to B and C."""
    event = participant.d.event
    A = "A" ^ event.A ** event.A
    B = "B" ^ event.B ** event.B
    C = "C" ^ event.C ** event.C

    rules = [
        "B_if_A" ^ A >> B,  # A → B
        "C_if_A" ^ A >> C,  # A → C
    ]
    participant.luk.rules.compile(*rules)
    return [A, B, C]


def init_knowledge_diamond_disjunction(participant: Participant) -> List[Chunk]:
    """Classic diamond with disjunctive convergence on D."""
    event = participant.d.event
    A = "A" ^ event.A ** event.A
    B = "B" ^ event.B ** event.B
    C = "C" ^ event.C ** event.C
    D = "D" ^ event.D ** event.D

    rules = [
        "B_if_A" ^ A >> B,  # A → B
        "C_if_A" ^ A >> C,  # A → C
        "D_if_B" ^ B >> D,  # B → D
        "D_if_C" ^ C >> D,  # C → D
    ]
    participant.luk.rules.compile(*rules)
    return [A, B, C, D]

def init_knowledge_spoon_disjunction(participant: Participant) -> List[Chunk]:
    """Classic diamond with disjunctive convergence on D."""
    event = participant.d.event
    A = "A" ^ event.A ** event.A
    B = "B" ^ event.B ** event.B
    C = "C" ^ event.C ** event.C
    D = "D" ^ event.D ** event.D
    E = "E" ^ event.E ** event.E
    F = "F" ^ event.F ** event.F

    rules = [
        "B_if_A" ^ A >> B,  # A → B
        "C_if_A" ^ A >> C,  # A → C
        "D_if_B" ^ B >> D,  # B → D
        "D_if_C" ^ C >> D,  # C → D
        "E_if_D" ^ D >> E,  # D → E
        "F_if_E" ^ E >> F,  # E → F
    ]
    participant.luk.rules.compile(*rules)
    return [A, B, C, D, E, F]

def init_knowledge_ladder(participant: Participant) -> List[Chunk]:
    """A ladder structure A → B → C → D."""
    event = participant.d.event
    A = "A" ^ event.A ** event.A
    B = "B" ^ event.B ** event.B
    C = "C" ^ event.C ** event.C
    D = "D" ^ event.D ** event.D

    rules = [
        "B_if_A" ^ A >> B,  # A → B
        "C_if_A" ^ A >> C,  # A → C
        "C_if_D" ^ D >> C,  # D → C
    ]
    participant.luk.rules.compile(*rules)
    return [A, B, C, D]

def init_knowledge_ladder_plus(participant: Participant) -> List[Chunk]:
    """A ladder structure A → B → C → D with extra links."""
    event = participant.d.event
    A = "A" ^ event.A ** event.A
    B = "B" ^ event.B ** event.B
    C = "C" ^ event.C ** event.C
    D = "D" ^ event.D ** event.D
    E = "E" ^ event.E ** event.E
    F = "F" ^ event.F ** event.F

    rules = [
        "B_if_A" ^ A >> B,  # A → B
        "C_if_A" ^ A >> C,  # A → C
        "C_if_D" ^ D >> C,  # D → C
        "F_if_E" ^ E >> F,  # E → F
    ]
    participant.luk.rules.compile(*rules)
    return [A, B, C, D, E, F]

def init_knowledge_net(participant: Participant) -> List[Chunk]:
    """A ladder structure A → B → C → D with extra links."""
    event = participant.d.event
    A = "A" ^ event.A ** event.A
    B = "B" ^ event.B ** event.B
    C = "C" ^ event.C ** event.C
    D = "D" ^ event.D ** event.D
    E = "E" ^ event.E ** event.E
    F = "F" ^ event.F ** event.F

    rules = [
        "B_if_A" ^ A >> B,  # A → B
        "C_if_A" ^ A >> C,  # A → C
        "E_if_B" ^ B >> E,  # B → E
        "E_if_C" ^ C >> E,  # C → E
        "C_if_D" ^ D >> C,  # D → C
        "F_if_C" ^ C >> F,  # C → F
    ]
    participant.luk.rules.compile(*rules)
    return [A, B, C, D, E, F]

def init_knowledge_diamond_conjunction(participant: Participant) -> List[Chunk]:
    """Diamond where B AND C are required to activate D."""
    event = participant.d.event
    A = "A" ^ event.A ** event.A
    B = "B" ^ event.B ** event.B
    C = "C" ^ event.C ** event.C
    D = "D" ^ event.D ** event.D

    rules = [
        "B_if_A" ^ A >> B,              # A → B
        "C_if_A" ^ A >> C,              # A → C
        "D_if_B_AND_C" ^ (B, C) >> D,   # B ∧ C → D
    ]
    participant.luk.rules.compile(*rules)
    return [A, B, C, D]


def init_knowledge_chain_three(participant: Participant) -> List[Chunk]:
    """A → B → C chain."""
    event = participant.d.event
    A = "A" ^ event.A ** event.A
    B = "B" ^ event.B ** event.B
    C = "C" ^ event.C ** event.C

    rules = [
        "B_if_A" ^ A >> B,  # A → B
        "C_if_B" ^ B >> C,  # B → C
    ]
    participant.luk.rules.compile(*rules)
    return [A, B, C]


def init_knowledge_chain_four(participant: Participant) -> List[Chunk]:
    """A → B → C → D chain."""
    event = participant.d.event
    A = "A" ^ event.A ** event.A
    B = "B" ^ event.B ** event.B
    C = "C" ^ event.C ** event.C
    D = "D" ^ event.D ** event.D

    rules = [
        "B_if_A" ^ A >> B,  # A → B
        "C_if_B" ^ B >> C,  # B → C
        "D_if_C" ^ C >> D,  # C → D
    ]
    participant.luk.rules.compile(*rules)
    return [A, B, C, D]

def init_knowledge_tall_glass(participant: Participant) -> List[Chunk]:
    """A → B → C → D chain."""
    event = participant.d.event
    A = "A" ^ event.A ** event.A
    B = "B" ^ event.B ** event.B
    C = "C" ^ event.C ** event.C
    D = "D" ^ event.D ** event.D
    E = "E" ^ event.E ** event.E

    rules = [
        "B_if_A" ^ A >> B,  # A → B
        "E_if_D" ^ D >> E,  # D → E
        "C_if_B" ^ B >> C,  # B → C
        "C_if_E" ^ E >> C,  # E → C
    ]
    participant.luk.rules.compile(*rules)
    return [A, B, C, D, E]

def init_knowledge_highway_entrance(participant: Participant) -> List[Chunk]:
    """highway entrance"""
    event = participant.d.event
    A = "A" ^ event.A ** event.A
    B = "B" ^ event.B ** event.B
    C = "C" ^ event.C ** event.C
    D = "D" ^ event.D ** event.D
    E = "E" ^ event.E ** event.E

    rules = [
        "B_if_A" ^ A >> B,  # A → B
        "C_if_B" ^ B >> C,  # B → C
        "C_if_E" ^ E >> C,  # E → C
        "B_if_D" ^ D >> B,  # D → B
        
    ]
    participant.luk.rules.compile(*rules)
    return [A, B, C, D, E]

def init_knowledge_crossroads(participant: Participant) -> List[Chunk]:
    """crossroads"""
    event = participant.d.event
    A = "A" ^ event.A ** event.A
    B = "B" ^ event.B ** event.B
    C = "C" ^ event.C ** event.C
    D = "D" ^ event.D ** event.D
    E = "E" ^ event.E ** event.E

    rules = [
        "B_if_A" ^ A >> B,  # A → B
        "D_if_A" ^ A >> D,  # A → D
        "C_if_B" ^ B >> C,  # B → C
        "E_if_D" ^ D >> E,  # D → E
        
    ]
    participant.luk.rules.compile(*rules)
    return [A, B, C, D, E]

def init_knowledge_bigworm_small_worm(participant: Participant) -> List[Chunk]:
    """big worm small worm"""
    event = participant.d.event
    A = "A" ^ event.A ** event.A
    B = "B" ^ event.B ** event.B
    C = "C" ^ event.C ** event.C
    D = "D" ^ event.D ** event.D
    E = "E" ^ event.E ** event.E

    rules = [
        "B_if_A" ^ A >> B,  # A → B
        "C_if_B" ^ B >> C,  # B → C
        "E_if_D" ^ D >> E,  # D → E
    ]
    participant.luk.rules.compile(*rules)
    return [A, B, C, D, E]

def init_knowledge_two_pyramids(participant: Participant) -> List[Chunk]:
    """A → B → C → D chain."""
    event = participant.d.event
    A = "A" ^ event.A ** event.A
    B = "B" ^ event.B ** event.B
    C = "C" ^ event.C ** event.C
    D = "D" ^ event.D ** event.D
    E = "E" ^ event.E ** event.E

    rules = [
        "B_if_A" ^ A >> B,  # A → B
        "C_if_B" ^ B >> C,  # B → C
        "D_if_A" ^ A >> D,  # A → D
        "E_if_B" ^ B >> E,  # B → E
    ]
    participant.luk.rules.compile(*rules)
    return [A, B, C, D, E]