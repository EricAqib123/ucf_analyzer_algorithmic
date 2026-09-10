import math


class UCFEngine:
    """
    Unified Complexity Framework (UCF) Engine
    ==========================================
    Original UCF formula:
        f(u) = f(t) + p(S) - p(D)

    Where:
        f(t)  = execution time
        p(S)  = Stability probability   — in [0, 1]
        p(D)  = Determinism probability — in [0, 1]
        TC    = Total Complexity        = f(t) + g(M)/1000
        R     = Remainder               = (1-p(S)) + (1-p(D)) + g(M)/1000

    Interpretation of f(u):
        Lower f(u) = faster and more deterministic (better / less complex)
        Higher f(u) = slower or less stable (more complex)
    """

    EPSILON = 1e-9

    # ------------------------------------------------------------------
    # p(S) — Stability Probability
    # ------------------------------------------------------------------
    # How stable/consistent the algorithm is relative to problem size.
    # Many steps completed quickly -> high stability -> near 1.
    # Output: always in [0, 1]
    #
    #   p(S) = exp( -time / steps )
    # ------------------------------------------------------------------
    def compute_pS(self, steps, time):
        steps = max(steps, 1)
        time  = max(time,  self.EPSILON)
        return float(math.exp(-time / steps))

    # ------------------------------------------------------------------
    # p(D) — Determinism Probability
    # ------------------------------------------------------------------
    # How reproducible/deterministic the output is.
    # High p(S) -> low runtime variation -> p(D) stays near 1.
    # Output: always in [0, 1]
    #
    #   p(D) = exp( -f(t) x (1 - p(S)) )
    # ------------------------------------------------------------------
    def compute_pD(self, ft, pS):
        ft  = max(ft,  self.EPSILON)
        pS  = max(min(pS, 1.0), 0.0)
        return float(math.exp(-ft * (1.0 - pS)))

    # ------------------------------------------------------------------
    # f(u) — UCF Score  <- ORIGINAL FORMULA
    # ------------------------------------------------------------------
    #   f(u) = f(t) + p(S) - p(D)
    #
    # Lower is better:
    #   Small f(t)  -> fast algorithm
    #   p(S) ~ p(D) -> stable and deterministic -> f(u) ~ f(t) (minimal)
    #   p(D) > p(S) -> very deterministic -> reduces f(u) further
    # ------------------------------------------------------------------
    def compute_fu(self, ft, pS, pD):
        ft  = max(ft,  self.EPSILON)
        pS  = max(min(pS, 1.0), 0.0)
        pD  = max(min(pD, 1.0), 0.0)
        return float(ft + pS - pD)

    # ------------------------------------------------------------------
    # TC — Total Complexity
    # ------------------------------------------------------------------
    #   TC(A,D) = f(t) + g(M) / 1000
    #
    # Memory divided by 1000 to normalise MB to same scale as seconds.
    # ------------------------------------------------------------------
    def compute_TC(self, ft, memory):
        ft     = max(ft,     0.0)
        memory = max(memory, 0.0)
        return float(ft + memory / 1000.0)

    # ------------------------------------------------------------------
    # R — Remainder Complexity
    # ------------------------------------------------------------------
    # Residual unexplained complexity: instability + non-determinism + memory.
    #
    #   R = (1 - p(S)) + (1 - p(D)) + g(M) / 1000
    # ------------------------------------------------------------------
    def compute_R(self, pS, pD, memory):
        pS     = max(min(pS, 1.0), 0.0)
        pD     = max(min(pD, 1.0), 0.0)
        memory = max(memory, 0.0)
        return float((1.0 - pS) + (1.0 - pD) + memory / 1000.0)