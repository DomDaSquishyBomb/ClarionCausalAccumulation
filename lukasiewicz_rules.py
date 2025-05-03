from pyClarion import (Process, Site, Atom, Sort, Family, keyform, ks_root, 
    Rule, Event, Priority, KeyForm, ChunkStore, Rules, UpdateSort, RuleStore)
from datetime import timedelta
import logging



class RuleStore2(RuleStore):

    rules: Rules
    lhs: ChunkStore
    rhs: ChunkStore
    main: Site
    riw: Site
    lhw: Site
    rhw: Site

    def __init__(self, 
                 name: str, 
                 r: Family,
                 c: Family, 
                 d: Family | Sort | Atom, 
                 v: Family | Sort, 
                 ) -> None:
        Process.__init__(self, name) # Disgusting... but works
        self.system.check_root(r, d, v)
        self.rules = Rules(); r[name] = self.rules
        with self:
            self.lhs = ChunkStore(f"{name}.chunks", c, d, v)
            self.rhs = self.lhs
        idx_r = self.system.get_index(keyform(self.rules))
        idx_lhs = self.system.get_index(keyform(self.lhs.chunks))
        idx_rhs = self.system.get_index(keyform(self.rhs.chunks))
        self.main = Site(idx_r, {}, c=0.0)
        self.riw = Site(idx_r * idx_r, {}, c=float("nan"))
        self.lhw = Site(idx_r * idx_lhs, {}, c=float("nan"))
        self.rhw = Site(idx_r * idx_rhs, {}, c=float("nan"))

    def resolve(self, event: Event) -> None:
        if event.source == self.lhs.bu.update:
            self.update()
        if event.source == self.compile:
            if self.system.logger.isEnabledFor(logging.DEBUG):
                self.log_compilation(event)
            ud_lhs, ud_rules = event.updates
            assert isinstance(ud_lhs, UpdateSort)
            assert isinstance(ud_rules, UpdateSort)
            self.lhs.compile_weights(*ud_lhs.add)
            self.compile_weights(*ud_rules.add)

    def compile(self, *rules: Rule, 
        dt: timedelta = timedelta(),
        priority: int = Priority.LEARNING
    ) -> None:
        """Encode a collection of new rules."""
        new_rules = []
        new_chunks = []
        for rule in rules:
            for chunk in rule._chunks_:
                chunk_instances = list(chunk._instantiations_())
                chunk._instances_.update(chunk_instances)
                if chunk not in new_chunks \
                    and ks_root(chunk) != self.system.root:
                    new_chunks.append(chunk)
            rule_instances = list(rule._instantiations_())
            rule._instances_.update(rule_instances)
            new_rules.append(rule)
            new_rules.extend(rule_instances)
        self.system.schedule(
            self.compile, 
            UpdateSort(self.lhs.chunks, add=tuple(new_chunks)),
            UpdateSort(self.rules, add=tuple(new_rules)),
            dt=dt, priority=priority)


class LukasiewiczRules(Process):
    main: Site
    input: Site
    strengths: Site
    rules: RuleStore2
    by: KeyForm

    def __init__(self, 
                 name: str, 
                 r: Family,
                 c: Family, 
                 d: Family | Sort | Atom, 
                 v: Family | Sort,
                 ) -> None:
        super().__init__(name)
        with self:
            self.rules = RuleStore2(f"{name}.rules", r, c, d, v)
        self.main = Site(self.rules.lhs.bu.main.index, {}, 0.0)
        self.input = Site(self.rules.lhs.bu.main.index, {}, 0.0)
        self.strengths = Site(self.rules.main.index, {}, 0.0)
        self.mul_by = keyform(self.rules.rules).agg * keyform(self.rules.rules)
        self.sum_by = keyform(self.rules.rules) * keyform(self.rules.rules).agg

    def resolve(self, event: Event) -> None:
        updates = [ud for ud in event.updates if isinstance(ud, Site.Update)]
        if self.input.affected_by(*updates):
            self.update()

    def update(self, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> None:
        bias = (self.rules.lhw[0]
                .abs()
                .sum(by=self.rules.main.index.kf)
                .shift(x=-1))
        rule_activations = (self.rules.lhw[0]
            .mul(self.input[0])
            .sum(by=self.rules.main.index.kf)
            .sub(bias)
            .bound_min(x=0.0)
            .with_default(c=self.main.const))
        strengths = (self.rules.riw[0]
            .mul(rule_activations, by=self.mul_by)
            .sum(by=self.sum_by)
            .with_default(c=self.main.const))
        main = (self.rules.rhw[0]
            .mul(rule_activations)
            .max(by=self.rules.rhs.td.input.index.kf)
            .with_default(c=self.rules.rhs.td.input.const))
        self.system.schedule(self.update, 
            self.main.update(main),
            self.strengths.update(strengths),
            dt=dt, priority=priority)