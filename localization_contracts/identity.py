"""Two rovers, one marker: keep the marker's track on the rover that carries it.

The marker is bound to the tag_rover track; the opponent track takes what is
left.  Continuation sources (lidar, silhouettes, flow) cannot tell two rovers
apart, so when the rovers pass close to each other the two tracks can leave
the encounter on each other's rover.  The marker is the only thing that can
notice: if, shortly after an encounter, it is seen where the opponent track is
and not where the tag_rover track is, the tracks went the wrong way, and their
states are exchanged -- each filter keeps its role, the state moves.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# What belongs to the estimate (moves on a swap) as opposed to the role of the
# filter (identity rules, gates, ages -- stays).
_STATE_FIELDS = ("xs", "Ps", "mu", "stamp_ns", "last_measurement_ns", "sources",
                 "history", "applied")


def swap_tracks(first, second):
    """Exchange the estimates of two ImmRoverFilter objects in place."""
    for name in _STATE_FIELDS:
        a, b = getattr(first, name), getattr(second, name)
        setattr(first, name, b)
        setattr(second, name, a)


@dataclass
class TwoRoverIdentity:
    """Encounter bookkeeping and the marker-on-the-wrong-track check."""

    close_m: float = 1.0          # an encounter: tracks nearer than this
    swap_gate_m: float = 0.5      # marker this near one track and not the other
    watch_s: float = 3.0          # how long after an encounter to keep checking
    events: list = field(default_factory=list)
    encounters: int = 0
    swaps: int = 0
    _inside: bool = False
    _started_ns: int | None = None
    _closest_m: float = float("inf")
    _watch_until_ns: int | None = None

    def observe_tracks(self, stamp_ns, tag_xy, opponent_xy):
        """Feed both published positions (None while a track is absent)."""
        if tag_xy is None or opponent_xy is None:
            return
        distance = float(np.hypot(tag_xy[0] - opponent_xy[0], tag_xy[1] - opponent_xy[1]))
        if distance < self.close_m:
            if not self._inside:
                self._inside = True
                self._started_ns = int(stamp_ns)
                self._closest_m = distance
                self.encounters += 1
            self._closest_m = min(self._closest_m, distance)
        elif self._inside:
            self._inside = False
            self._watch_until_ns = int(stamp_ns + self.watch_s * 1e9)
            self.events.append({"event": "encounter", "start_ns": self._started_ns,
                                "end_ns": int(stamp_ns),
                                "closest_m": round(self._closest_m, 3)})

    def watching(self, stamp_ns):
        return self._inside or (self._watch_until_ns is not None
                                and stamp_ns <= self._watch_until_ns)

    def marker_says_swap(self, stamp_ns, marker_xy, tag_xy, opponent_xy):
        """True when a fresh marker sits on the opponent track, not on its own."""
        if marker_xy is None or tag_xy is None or opponent_xy is None:
            return False
        if not self.watching(stamp_ns):
            return False
        to_tag = float(np.hypot(marker_xy[0] - tag_xy[0], marker_xy[1] - tag_xy[1]))
        to_opponent = float(np.hypot(marker_xy[0] - opponent_xy[0], marker_xy[1] - opponent_xy[1]))
        return to_opponent < self.swap_gate_m and to_tag > self.swap_gate_m

    def swap(self, stamp_ns, tag_filter, opponent_filter, marker_xy):
        swap_tracks(tag_filter, opponent_filter)
        self.swaps += 1
        self.events.append({"event": "swap", "stamp_ns": int(stamp_ns),
                            "marker_xy": [float(marker_xy[0]), float(marker_xy[1])]})

    def summary(self):
        return {"encounters_closer_than_m": self.close_m, "encounters": self.encounters,
                "swaps": self.swaps, "events": list(self.events)}
