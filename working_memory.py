from datetime import timedelta
from pyClarion import Event, Priority, Site, NumDict
from pyClarion.components import Pool


class WorkingMemory(Pool):
    gated_output: Site

    def __init__(
        self,
        name: str,
        p,
        s,
        *,
        func=NumDict.sum,
        inhibition_strength: float = 0.0,
        threshold: float = 0.1,
        **kwargs
    ) -> None:
        super().__init__(name, p, s, func=func, **kwargs)

        # Store gating parameters
        self.inhibition_strength = inhibition_strength
        self.threshold = threshold

        # Will be set externally to forward_inference.main.index
        self.gated_output = None
        self._luk_process = None  # Reference to forward inference (set externally)

    def set_gating_target(self, forward_inference) -> None:
        """
        Configure rule gating for a specific forward inference process.

        Parameters
        ----------
        forward_inference : ForwardInference
            The forward inference process whose outputs will be gated.
        """
        self._luk_process = forward_inference

        # Create gated output site with same index as forward inference output
        self.gated_output = Site(
            forward_inference.main.index, {}, c=0.0
        )

    def resolve(self, event: Event) -> None:
        """
        Trigger updates when inputs change or when forward inference updates.
        """
        # Call parent Pool's resolve to handle normal pooling
        super().resolve(event)

        # Also update gating when forward inference fires (if configured)
        if self._luk_process is not None:
            if event.source == self._luk_process.update:
                self.update_gating()

            # Also update gating when WM contents change
            updates = [ud for ud in event.updates if isinstance(ud, Site.Update)]
            if self.main.affected_by(*updates):
                self.update_gating()

    def update_gating(
        self,
        dt: timedelta = timedelta(),
        priority: int = Priority.PROPAGATION,
    ) -> None:
        """
        Gate rule outputs based on current WM contents.

        Strategy:
        1. Identify which LHS chunks are active in WM (above threshold)
        2. For each rule, check if ANY of its LHS factors are in WM
        3. If yes: allow rule output (multiply by 1.0)
        4. If no: inhibit rule output (multiply by inhibition_strength)
        """
        if self._luk_process is None or self.gated_output is None:
            return  # Not configured yet

        # Get current WM activations
        wm_activations = self.main[0]

        # Get current forward inference outputs
        luk_outputs = self._luk_process.main[0]

        # Identify which chunks are active in WM above threshold
        wm_active = (
            wm_activations
            .bound_min(x=self.threshold)
            .bound_max(x=1.0)  # Binarize to 0 or 1
            .with_default(c=0.0)
        )

        # Get left-hand weights (maps rules � LHS chunks)
        lhw = self._luk_process.rules.lhw[0]

        # For each rule, determine if ANY of its LHS factors are in WM
        # Strategy: multiply lhw connections by wm_active, then take max per rule
        lhs_in_wm = (
            lhw
            .abs()  # Get absolute weights to identify connections
            .bound_min(x=0.01)  # Keep only meaningful connections
            .mul(wm_active)  # Weight by WM
            .max(by=self._luk_process.rules.main.index.kf)  # Max across LHS for each rule
            .bound_max(x=1.0)  # Normalize to binary [0, 1]
            .with_default(c=0.0)
        )

        # Get right-hand weights (maps rules � RHS chunks)
        rhw = self._luk_process.rules.rhw[0]

        # Propagate rule-level gating to output chunks
        # For each output chunk, take the max gate value across rules that produce it
        output_gate = (
            rhw
            .abs()  # Get absolute weights to identify connections
            .bound_min(x=0.01)  # Keep only meaningful connections
            .mul(lhs_in_wm, by=self._luk_process.rules.main.index.kf)  # Apply rule gates
            .max(by=self._luk_process.main.index.kf)  # Max across rules for each output
            .with_default(c=0.0)
        )

        # Create final modulation factors
        # If gate=1 (LHS in WM): factor = 1.0
        # If gate=0 (LHS not in WM): factor = inhibition_strength
        modulation_factor = (
            output_gate
            .scale(x=1.0 - self.inhibition_strength)  # Scale: (1-s) or 0
            .shift(x=self.inhibition_strength)  # Shift: 1 or s
            .with_default(c=self.inhibition_strength)
        )

        # Apply modulation to forward inference outputs
        modulated = luk_outputs.mul(modulation_factor).with_default(c=0.0)

        # Schedule update
        self.system.schedule(
            self.update_gating,
            self.gated_output.update(modulated),
            dt=dt,
            priority=priority,
        )
