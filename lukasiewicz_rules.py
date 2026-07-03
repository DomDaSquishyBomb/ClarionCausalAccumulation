"""
lukasiewicz_rules.py

Custom rule-store process (`RuleStore2`) plus a Forward Łukasiewicz-logic rule layer
(`ForwardInference`) for use inside the SCR agent.

Generously donated to me by Can Mekik.
"""

from pyClarion import (Process, Site, Atom, Sort, Family, keyform, ks_root, 
    Rule, Event, Priority, KeyForm, ChunkStore, Rules, UpdateSort, RuleStore, numdict)
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


class ForwardInference(Process):
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
        
        # funny Dominic Le stuff
        
        if self.rules.lhs.chunks is not self.rules.rhs.chunks:
            raise ValueError("LHS and RHS chunk stores must be identical.")
        
        rules = self.rules.rules
        chunks = self.rules.lhs.chunks
        
        idx_r = self.system.get_index(keyform(rules))
        idx_c = self.system.get_index(keyform(chunks))

        self.site_c = Site(idx_c, {}, c=1.0)
        self.site_rcc = Site(idx_r * idx_c * idx_c, {}, c=1.0)
        self.site_ccc = Site(idx_c * idx_c * idx_c, {}, c=1.0)
        
        self.by_rxc = keyform(rules) * keyform(chunks).agg *  keyform(chunks)
        self.by_rcx = keyform(rules) * keyform(chunks) * keyform(chunks).agg
        self.by_xcc = keyform(rules).agg * keyform(chunks) * keyform(chunks)
        
        self.by_ccx = keyform(chunks) * keyform(chunks) * keyform(chunks).agg
        self.by_xcc = keyform(chunks).agg * keyform(chunks) * keyform(chunks)
        self.by_cxc = keyform(chunks) * keyform(chunks).agg * keyform(chunks)
        
        self.by_cx = keyform(chunks) * keyform(chunks).agg
        
        self.by_xc = keyform(chunks).agg * keyform(chunks)
        
        self.by_xx = keyform(chunks).agg * keyform(chunks).agg
        
        self.by_cc = keyform(chunks) * keyform(chunks)
        
        self.by_rc = keyform(rules) * keyform(chunks)
        
        self.by_c = keyform(chunks)
        
        

    def resolve(self, event: Event) -> None:
        updates = [ud for ud in event.updates if isinstance(ud, Site.Update)]
        if self.input.affected_by(*updates):
            self.update()

    def update(self,
        dt: timedelta = timedelta(),
        priority: int = Priority.PROPAGATION
    ) -> None:
        
        #w = self.site_rcc[0].mul(self.rules.lhw[0].with_default(c=0.0), by = (self.by_rxc)).mul(self.rules.rhw[0].with_default(c=0.0), by = (self.by_rcx)).sum(by = (self.by_xcc)) 
        #wwt = self.site_ccc[0].mul(w, by=(self.by_cxc)).mul(w, by=(self.by_ccx)).sum(by=(self.by_xcc)).neg()
        #wwtc = wwt.mul(self.input[0], by=(self.by_xc)).sum(by=(self.by_cx))
        #wwtc2 = wwtc.scale(x=0.5)
        
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



class BackwardInference(Process):
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
        self.main = Site(self.rules.rhs.bu.main.index, {}, 0.0)
        self.input = Site(self.rules.rhs.bu.main.index, {}, 0.0)
        self.strengths = Site(self.rules.main.index, {}, 0.0)
        self.mul_by = keyform(self.rules.rules).agg * keyform(self.rules.rules)
        self.sum_by = keyform(self.rules.rules) * keyform(self.rules.rules).agg
        
        # funny Dominic Le stuff
        
        if self.rules.lhs.chunks is not self.rules.rhs.chunks:
            raise ValueError("LHS and RHS chunk stores must be identical.")
        
        rules = self.rules.rules
        chunks = self.rules.lhs.chunks
        
        idx_r = self.system.get_index(keyform(rules))
        idx_c = self.system.get_index(keyform(chunks))

        self.site_c = Site(idx_c, {}, c=1.0)
        self.site_rcc = Site(idx_r * idx_c * idx_c, {}, c=1.0)
        self.site_ccc = Site(idx_c * idx_c * idx_c, {}, c=1.0)
        
        self.by_rxc = keyform(rules) * keyform(chunks).agg *  keyform(chunks)
        self.by_rcx = keyform(rules) * keyform(chunks) * keyform(chunks).agg
        self.by_xcc = keyform(rules).agg * keyform(chunks) * keyform(chunks)
        
        self.by_ccx = keyform(chunks) * keyform(chunks) * keyform(chunks).agg
        self.by_xcc = keyform(chunks).agg * keyform(chunks) * keyform(chunks)
        self.by_cxc = keyform(chunks) * keyform(chunks).agg * keyform(chunks)
        
        self.by_cx = keyform(chunks) * keyform(chunks).agg
        
        self.by_xc = keyform(chunks).agg * keyform(chunks)
        
        self.by_xx = keyform(chunks).agg * keyform(chunks).agg
        
        self.by_cc = keyform(chunks) * keyform(chunks)
        
        self.by_rc = keyform(rules) * keyform(chunks)
        
        self.by_c = keyform(chunks)

    def resolve(self, event: Event) -> None:
        updates = [ud for ud in event.updates if isinstance(ud, Site.Update)]
        if self.input.affected_by(*updates):
            self.update()

    def update(self,
        dt: timedelta = timedelta(),
        priority: int = Priority.PROPAGATION
    ) -> None:
        
        w = self.site_rcc[0].mul(self.rules.lhw[0].with_default(c=0.0), by = (self.by_rxc)).mul(self.rules.rhw[0].with_default(c=0.0), by = (self.by_rcx)).sum(by = (self.by_xcc)) 
        
        wwt = self.site_ccc[0].mul(w, by=(self.by_cxc)).mul(w, by=(self.by_ccx)).sum(by=(self.by_xcc)).neg()
        
        wwtc = wwt.mul(self.input[0], by=(self.by_xc)).sum(by=(self.by_cx))
        
        wwtc2 = wwtc.scale(x=0.5)
        
        bias = (self.rules.rhw[0]
                .abs()
                .sum(by=self.rules.main.index.kf)
                .shift(x=-1))
        rule_activations = (self.rules.rhw[0]
            .mul(self.input[0])
            .sum(by=self.rules.main.index.kf)
            .sub(bias)
            .bound_min(x=0.0)
            .with_default(c=self.main.const))
        strengths = (self.rules.riw[0]
            .mul(rule_activations, by=self.mul_by)
            .sum(by=self.sum_by)
            .with_default(c=self.main.const))
        main = (self.rules.lhw[0]
            .mul(rule_activations)
            .max(by=self.rules.rhs.td.input.index.kf)
            .with_default(c=self.rules.rhs.td.input.const).sum(wwtc))
        self.system.schedule(self.update,
            self.main.update(main),
            self.strengths.update(strengths),
            dt=dt, priority=priority)

