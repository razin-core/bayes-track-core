# -*- coding: utf-8 -*-
"""
core/yager_dempster.py - Modified Yager Dempster-Shafer Evidence Fusion with
Source Base Overlap Index O_ij and Exponential 3-Minute Echo-Chamber Penalty.
"""
import config.settings  # Enforce single-thread BLAS
import math
from typing import Dict, List, Tuple, Optional, Any

from config.settings import ECHO_SUPPRESSION_WINDOW_SEC, ECHO_NOVELTY_DISCOUNT

# Pairwise Source Overlap Matrix O_ij in [0.0 .. 1.0]
# Reflects shared regional/national OSINT feeder channels between aggregators
SOURCE_OVERLAP_INDEX: Dict[Tuple[str, str], float] = {
    ("api_neptun_threats", "api_neptun_messages"): 0.85,
    ("api_neptun_messages", "tg_ppo_vinnytsia"): 0.75,
    ("api_neptun_messages", "tg_vinnytsiarealll"): 0.70,
    ("api_neptun_messages", "tg_vanek_nikolaev"): 0.65,
    ("api_neptun_messages", "tg_war_monitor"): 0.65,
    ("tg_ppo_vinnytsia", "tg_vinnytsiarealll"): 0.80,
    ("tg_vanek_nikolaev", "tg_war_monitor"): 0.60,
    ("api_mapa_ua", "api_neptun_threats"): 0.35,
    ("tg_kpszsu", "tg_vanek_nikolaev"): 0.30,
    ("tg_kpszsu", "api_mapa_ua"): 0.20,
}


def get_source_overlap(source_a: str, source_b: str) -> float:
    """Returns symmetric source feeder overlap index O_ij in [0.0 .. 1.0]."""
    if source_a == source_b:
        return 1.0
    if (source_a, source_b) in SOURCE_OVERLAP_INDEX:
        return SOURCE_OVERLAP_INDEX[(source_a, source_b)]
    if (source_b, source_a) in SOURCE_OVERLAP_INDEX:
        return SOURCE_OVERLAP_INDEX[(source_b, source_a)]
    # Default overlap between two Telegram regional monitors vs independent radar API
    if source_a.startswith("tg_") and source_b.startswith("tg_"):
        return 0.55
    return 0.15


def compute_echo_novelty_factor(
    dt_sec: float,
    source_curr: str,
    source_prev: str,
    same_h3_or_locality: bool = True,
) -> float:
    """
    Computes information novelty discount factor alpha_novelty in [0.20 .. 1.00].
    If a repost arrives within dt < 180 s for the same H3 cell/locality (SDD Section 3.3),
    applies an exponential penalty weighted by the source overlap index O_ij.
    """
    if not same_h3_or_locality or dt_sec >= ECHO_SUPPRESSION_WINDOW_SEC:
        return 1.00

    dt_clamped = max(0.0, float(dt_sec))
    o_ij = get_source_overlap(source_curr, source_prev)

    # For direct reposts in < 180 s with high overlap, novelty floor is ECHO_NOVELTY_DISCOUNT (0.20)
    # With low overlap (independent confirmation), exponential recovery is faster
    exp_decay = math.exp(-dt_clamped / 60.0)
    penalty = exp_decay * (0.65 + 0.35 * o_ij)
    alpha = max(ECHO_NOVELTY_DISCOUNT, 1.0 - (1.0 - ECHO_NOVELTY_DISCOUNT) * penalty)
    # Strict SDD 3.3 canonical rule when same locality repost arrives rapidly (< 90s) from overlapping channel
    if dt_clamped < 90.0 and o_ij >= 0.50:
        return ECHO_NOVELTY_DISCOUNT
    return round(float(alpha), 4)


class YagerDempsterFusion:
    """
    Modified Dempster-Shafer combination rule by Ronald R. Yager on frame
    Theta = {THREAT, CLEAR}.
    Routes conflict mass K into universal ignorance m(Theta) instead of normalizing
    by (1 - K), completely eliminating Zadeh's paradox and zero-division singularities.
    """

    @staticmethod
    def combine_two_bpa(
        m1: Tuple[float, float, float],
        m2: Tuple[float, float, float],
    ) -> Tuple[float, float, float, float]:
        """
        Combines two Basic Probability Assignments m = (m_threat, m_clear, m_theta).
        Returns (m12_threat, m12_clear, m12_theta, conflict_K).
        """
        t1, c1, u1 = m1
        t2, c2, u2 = m2

        # Conflict mass K (SDD Section 3.3)
        K = t1 * c2 + c1 * t2

        m12_threat = t1 * t2 + t1 * u2 + u1 * t2
        m12_clear = c1 * c2 + c1 * u2 + u1 * c2
        # In Yager's rule, conflict K is added to full ignorance m12(Theta)
        m12_theta = u1 * u2 + K

        # Numerical normalization guard so masses sum to 1.000000
        total = m12_threat + m12_clear + m12_theta
        if total > 0:
            m12_threat /= total
            m12_clear /= total
            m12_theta /= total

        return m12_threat, m12_clear, m12_theta, K

    @classmethod
    def fuse_evidence_sequence(
        cls,
        evidences: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Fuses a chronological sequence of source evidences:
        each item has: {'source_id': str, 'timestamp': float, 'base_trust': float,
                        'is_clear': bool, 'h3_cell': str}
        Returns fused belief mass and Mahalanobis ellipse expansion scale:
        R_ellipse_scale = 1.0 + 1.5 * K (SDD Section 3.3).
        """
        if not evidences:
            return {
                "m_threat": 0.0,
                "m_clear": 0.0,
                "m_theta": 1.0,
                "conflict_K": 0.0,
                "ellipse_scale": 1.0,
            }

        sorted_ev = sorted(evidences, key=lambda x: float(x.get("timestamp", 0.0)))
        curr_m: Tuple[float, float, float] = (0.0, 0.0, 1.0)
        cumulative_conflict = 0.0

        prev_ts: Optional[float] = None
        prev_src: Optional[str] = None
        prev_h3: Optional[str] = None

        for ev in sorted_ev:
            src = str(ev.get("source_id", "api_mapa_ua"))
            ts = float(ev.get("timestamp", 0.0))
            w_base = max(0.05, min(0.98, float(ev.get("base_trust", 0.75))))
            is_clear = bool(ev.get("is_clear", False))
            h3_cell = str(ev.get("h3_cell", ""))

            if prev_ts is not None and prev_src is not None:
                same_loc = (h3_cell == prev_h3) if (h3_cell and prev_h3) else True
                alpha = compute_echo_novelty_factor(ts - prev_ts, src, prev_src, same_loc)
            else:
                alpha = 1.0

            w_eff = w_base * alpha
            if is_clear:
                m_ev = (0.0, w_eff, 1.0 - w_eff)
            else:
                m_ev = (w_eff, 0.0, 1.0 - w_eff)

            t12, c12, u12, k_step = cls.combine_two_bpa(curr_m, m_ev)
            curr_m = (t12, c12, u12)
            cumulative_conflict = max(cumulative_conflict, k_step)

            prev_ts = ts
            prev_src = src
            prev_h3 = h3_cell

        ellipse_scale = 1.0 + 1.5 * cumulative_conflict
        return {
            "m_threat": round(curr_m[0], 4),
            "m_clear": round(curr_m[1], 4),
            "m_theta": round(curr_m[2], 4),
            "conflict_K": round(cumulative_conflict, 4),
            "ellipse_scale": round(ellipse_scale, 4),
        }
