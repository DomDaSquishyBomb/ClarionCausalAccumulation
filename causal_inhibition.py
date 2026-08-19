from lukasiewicz_rules import RuleStore2
from pyClarion import Process, Site, Event, Priority, Family, Sort, Atom, KeyForm, keyform
from datetime import timedelta


class CausalInhibition(Process):
    main: Site
    input: Site
    strengths: Site
    rules: RuleStore2
    by: KeyForm

    def __init__(self,
        name: str,
        p: Family,
        s: Family,
        ) -> None:
        super().__init__(name)
        self.main = Site(p, {}, c=0.0)
        self.input = Site(p, {}, c=0.0)
        self.rules = Site(s, {}, c=0.0)
        self.strengths = Site(s.index, {}, 0.0)


    def resolve(self, event: Event) -> None:
        updates = [ud for ud in event.updates if isinstance(ud, Site.Update)]
        if self.input.affected_by(*updates):
            self.update()

    def update(
        self,
        dt: timedelta = timedelta(),
        priority: int = Priority.PROPAGATION,
    ) -> None:
        
        