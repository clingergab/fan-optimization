"""Per-design cross-tier config-hash + §9.4.1 consistency assertion.

§9.4.1 requires that the **cross-tier locked numerics** are identical for a design
across every tier/eval it is run at. Those numerics live in
:data:`fanopt.cfd.configs.CROSS_TIER` (MACH is deliberately *not* there — it is
tier-specific per the Round-9 HIGH-12 lock). This module hashes that dict to a
stable ``config_hash`` and provides the assertion the orchestrator runs over a
design's ledger rows: any drift in a locked numeric flips the hash and trips it.
"""

from __future__ import annotations

from collections.abc import Iterable

from fanopt.cfd.configs import CROSS_TIER
from fanopt.utils.ledger import design_hash

__all__ = [
    "cross_tier_config_hash",
    "CROSS_TIER_CONFIG_HASH",
    "assert_config_hash",
    "assert_design_config_consistent",
]


def cross_tier_config_hash() -> str:
    """Stable 24-hex hash of the locked cross-tier numerics.

    Identical for every design and tier by construction; a different value means
    a cross-tier locked numeric has drifted.
    """
    return design_hash(dict(CROSS_TIER))


CROSS_TIER_CONFIG_HASH: str = cross_tier_config_hash()


def assert_config_hash(hash_value: str) -> None:
    """Raise if ``hash_value`` is not the locked cross-tier hash (numerics drift)."""
    if hash_value != CROSS_TIER_CONFIG_HASH:
        raise ValueError(
            f"config_hash {hash_value!r} != locked cross-tier hash "
            f"{CROSS_TIER_CONFIG_HASH!r} (§9.4.1 numerics drift)"
        )


def assert_design_config_consistent(config_hashes: Iterable[str]) -> None:
    """§9.4.1: every eval of a design must carry the identical cross-tier hash.

    Raises if the sequence is empty or any hash differs from the locked value.
    """
    hs = list(config_hashes)
    if not hs:
        raise ValueError("no config hashes to check for cross-tier consistency")
    bad = sorted({h for h in hs if h != CROSS_TIER_CONFIG_HASH})
    if bad:
        raise ValueError(
            f"§9.4.1 cross-tier config drift: {bad} != locked {CROSS_TIER_CONFIG_HASH!r}"
        )
