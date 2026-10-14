"""PRELIMINARY liveness check: head-turn challenge. This is a heuristic, NOT a trained CNN.

The person must turn their head left AND right within LIVENESS_TIMEOUT_S. Yaw is estimated from
YuNet's 5 landmarks: horizontal offset of the nose from the eye midpoint, divided by eye distance.
A flat printed photo keeps that ratio roughly constant, so a static photo fails.
Known gap: a video replay of someone turning their head WILL pass. The planned CNN anti-spoofing
model (transfer learning, e.g. on NUAA / custom data) should replace or join this behind check().
"""
import time

import config


def yaw_ratio(face):
    rex, rey, lex, ley, nx = face[4], face[5], face[6], face[7], face[8]
    eye_mid = (rex + lex) / 2
    eye_dist = max(abs(lex - rex), 1.0)
    return (nx - eye_mid) / eye_dist


class HeadTurnChallenge:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()
        self.saw_left = self.saw_right = False
        self.passed = False

    def check(self, face):
        """Returns (state_text, passed)."""
        if config.LIVENESS_MODE == "bypass_test":
            return "LIVENESS BYPASSED (TEST)", True
        if self.passed:
            return "LIVE", True
        r = yaw_ratio(face)
        self.saw_left |= r < -config.LIVENESS_TURN_RATIO
        self.saw_right |= r > config.LIVENESS_TURN_RATIO
        if self.saw_left and self.saw_right:
            self.passed = True
            return "LIVE", True
        if time.time() - self.start > config.LIVENESS_TIMEOUT_S:
            self.reset()
            return "LIVENESS FAILED - retry", False
        # note: direction-agnostic prompt, so mirrored/unmirrored previews don't matter
        done = int(self.saw_left) + int(self.saw_right)
        return f"CHECKING LIVENESS: turn head side to side ({done}/2)", False
