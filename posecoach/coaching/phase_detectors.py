from __future__ import annotations
from dataclasses import dataclass

@dataclass
class PhaseEvent:
    name: str
    frame_idx: int

class SquatPhaseDetector:
    def __init__(self, up_enter=160.0, down_enter=150.0, bottom_enter=110.0, hysteresis=8.0):
        self.state="up"; self.up_enter=up_enter; self.down_enter=down_enter; self.bottom_enter=bottom_enter; self.h=hysteresis
    def update(self, frame_idx: int, knee_angle_deg: float):
        ev=[]; a=knee_angle_deg
        if self.state=="up" and a < (self.down_enter - self.h):
            self.state="down"; ev.append(PhaseEvent("squat_down_start", frame_idx))
        elif self.state=="down":
            if a < (self.bottom_enter + self.h):
                self.state="bottom"; ev.append(PhaseEvent("squat_bottom_reached", frame_idx))
            elif a > (self.up_enter + self.h):
                self.state="up"; ev.append(PhaseEvent("squat_up_start", frame_idx))
        elif self.state=="bottom" and a > (self.bottom_enter + self.h):
            self.state="up"; ev.append(PhaseEvent("squat_up_start", frame_idx))
        return ev

class JacksPhaseDetector:
    def __init__(self, up_enter=100.0, down_enter=40.0, hysteresis=8.0):
        self.state="down"; self.up_enter=up_enter; self.down_enter=down_enter; self.h=hysteresis
    def update(self, frame_idx: int, shoulder_abd_deg: float):
        ev=[]; a=shoulder_abd_deg
        if self.state=="down" and a > (self.up_enter + self.h):
            self.state="up"; ev.append(PhaseEvent("jacks_up_start", frame_idx))
        elif self.state=="up" and a < (self.down_enter - self.h):
            self.state="down"; ev.append(PhaseEvent("jacks_down_start", frame_idx))
        return ev
