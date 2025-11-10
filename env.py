# Minimal pass-and-score environment without external gym dependency.
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
import matplotlib.pyplot as plt

# Minimal spaces stubs
class Space:
    def sample(self, rng=None):
        raise NotImplementedError

class Discrete(Space):
    def __init__(self, n:int):
        self.n = int(n)
    def sample(self, rng=None):
        import numpy as _np
        rng = rng or _np.random.default_rng()
        return int(rng.integers(0, self.n))

class MultiDiscrete(Space):
    def __init__(self, nvec):
        import numpy as _np
        self.nvec = _np.array(nvec, dtype=int)
    def sample(self, rng=None):
        import numpy as _np
        rng = rng or _np.random.default_rng()
        return _np.array([rng.integers(0, n) for n in self.nvec], dtype=int)

class Box(Space):
    def __init__(self, low, high, shape=None, dtype=np.float32):
        import numpy as _np
        self.low = _np.full(shape, low, dtype=dtype) if _np.isscalar(low) else _np.array(low, dtype=dtype)
        self.high = _np.full(shape, high, dtype=dtype) if _np.isscalar(high) else _np.array(high, dtype=dtype)
        self.shape = shape or self.low.shape
        self.dtype = dtype
    def sample(self, rng=None):
        import numpy as _np
        rng = rng or _np.random.default_rng()
        return rng.uniform(self.low, self.high).astype(self.dtype)

@dataclass
class Region:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    def sample_point(self, rng: np.random.Generator, interior_ratio: float = 1.0) -> np.ndarray:
        p = float(interior_ratio)
        if p <= 0.0:
            raise ValueError("interior_ratio must be > 0.0")
        if p > 1.0:
            raise ValueError("interior_ratio must be <= 1.0")

        cx = (self.xmin + self.xmax) / 2.0
        cy = (self.ymin + self.ymax) / 2.0
        half_w = (self.xmax - self.xmin) * 0.5 * interior_ratio
        half_h = (self.ymax - self.ymin) * 0.5 * interior_ratio

        xmin = cx - half_w
        xmax = cx + half_w
        ymin = cy - half_h
        ymax = cy + half_h

        return np.array([
            rng.uniform(xmin, xmax),
            rng.uniform(ymin, ymax)
        ], dtype=np.float32)

    def contains_point(self, p: np.ndarray) -> bool:
        return (self.xmin <= p[0] <= self.xmax) and (self.ymin <= p[1] <= self.ymax)

class PassAndScoreEnv:
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        centralized: bool = True,
        seed: Optional[int] = None,
        dt: float = 0.05,
        max_steps: int = 400,
        field_padding: float = 0.2,
        region_a: Optional[Region] = None,
        region_b: Optional[Region] = None,
        goal_width: float = 0.4,
        agent_radius: float = 0.05,
        ball_radius: float = 0.03,
        max_speed_agent: float = 1.2,
        max_speed_ball: float = 2.0,
        accel: float = 1.0,
        friction_agent: float = 0.95,
        friction_ball: float = 0.995,
        restitution: float = 0.5,
        sample_interior_ratio: float = 0.8,
    ):
        self.centralized = centralized
        self.dt = dt
        self.max_steps = max_steps
        self.field_padding = field_padding
        self.region_a = region_a or Region(-2.0, -0.2, -1.0, 1.0)
        self.region_b = region_b or Region(0.2, 2.0, -1.0, 1.0)
        self.goal_width = goal_width

        self.agent_radius = agent_radius
        self.ball_radius = ball_radius
        self.max_speed_agent = max_speed_agent
        self.max_speed_ball = max_speed_ball
        self.accel = accel
        self.friction_agent = friction_agent
        self.friction_ball = friction_ball
        self.restitution = restitution

        self.np_random: np.random.Generator = np.random.default_rng(seed)
        self.sample_interior_ratio = sample_interior_ratio

        self.goal_y = self.region_b.ymax
        gx_center = (self.region_b.xmin + self.region_b.xmax) / 2.0
        self.goal_xmin = gx_center - self.goal_width / 2.0
        self.goal_xmax = gx_center + self.goal_width / 2.0

        # actions: 0 stay, 1 up, 2 down, 3 left, 4 right
        self.single_action_space = Discrete(5)
        self.action_space = MultiDiscrete([5, 5])

        # Observations:
        # centralized: pA(2), pB(2), vA(2), vB(2), pball(2), vball(2) -> 12
        self.observation_space_central = Box(-np.inf, np.inf, shape=(12,), dtype=np.float32)
        # per-agent (decentralized): own p(2), own v(2), rel p ball(2), rel v ball(2), rel p other(2), rel v other(2) -> 12
        self.observation_space_agent = Box(-np.inf, np.inf, shape=(12,), dtype=np.float32)

        self.state: Dict[str, Any] = {}
        self.fig = None
        self.ax = None

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed(seed)

        pA = self.region_a.sample_point(self.np_random, interior_ratio=self.sample_interior_ratio)
        pB = self.region_b.sample_point(self.np_random, interior_ratio=self.sample_interior_ratio)
        vA = np.zeros(2, dtype=np.float32)
        vB = np.zeros(2, dtype=np.float32)
        pball = self.region_b.sample_point(self.np_random, interior_ratio=self.sample_interior_ratio)
        vball = np.zeros(2, dtype=np.float32)

        # NEW/CHANGED: track last player to touch the ball; default "A" (so shaping can default to pass-to-B)
        self.state = dict(
            pA=pA, pB=pB, vA=vA, vB=vB,
            pball=pball, vball=vball, steps=0,
            last_touch="A"  # default as requested
        )

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: Union[np.ndarray, Tuple[int, int]]):
        aA, aB = int(action[0]), int(action[1])
        s = self.state
        info = {"before": self._get_info()}

        # action -> direction vector mapping
        def action_vec(a: int) -> np.ndarray:
            if a == 1:   # up
                return np.array([0.0, 1.0], dtype=np.float32)
            elif a == 2: # down
                return np.array([0.0, -1.0], dtype=np.float32)
            elif a == 3: # left
                return np.array([-1.0, 0.0], dtype=np.float32)
            elif a == 4: # right
                return np.array([1.0, 0.0], dtype=np.float32)
            else:        # stay / unknown
                return np.array([0.0, 0.0], dtype=np.float32)

        s["vA"] = s["vA"] + self.accel * self.dt * action_vec(aA)
        s["vB"] = s["vB"] + self.accel * self.dt * action_vec(aB)

        def cap(v, m):
            sp = np.linalg.norm(v)
            return v * (m / sp) if sp > m else v
        s["vA"] = cap(s["vA"], self.max_speed_agent)
        s["vB"] = cap(s["vB"], self.max_speed_agent)
        # don't pre-cap vball here; we'll cap after impulses / bounces below

        s["pA"] = s["pA"] + s["vA"] * self.dt
        s["pB"] = s["pB"] + s["vB"] * self.dt
        s["pball"] = s["pball"] + s["vball"] * self.dt

        # NEW/CHANGED: track last toucher when contact occurs
        def try_contact(p_agent, v_agent, who_label: str):
            d = s["pball"] - p_agent
            dist = float(np.linalg.norm(d))
            if dist < (self.agent_radius + self.ball_radius):
                vel_norm = np.linalg.norm(v_agent)
                if vel_norm > 1e-6:
                    dir_imp = v_agent / (vel_norm + 1e-12)
                else:
                    if dist > 1e-6:
                        dir_imp = d / (dist + 1e-12)
                    else:
                        dir_imp = np.array([0.0, 1.0], dtype=np.float32)
                impulse = 0.8 * dir_imp
                s["vball"] += impulse
                n = d / (dist + 1e-6) if dist > 0.0 else dir_imp
                s["pball"] = p_agent + n * (self.agent_radius + self.ball_radius + 1e-3)
                s["last_touch"] = who_label  # record last player to touch

        try_contact(s["pA"], s["vA"], "A")
        try_contact(s["pB"], s["vB"], "B")

        s["vA"] *= self.friction_agent
        s["vB"] *= self.friction_agent
        s["vball"] *= self.friction_ball

        # Bounce off world bounds (reflect velocity with restitution)
        xmin = self.region_a.xmin - self.field_padding
        xmax = self.region_b.xmax + self.field_padding
        ymin = min(self.region_a.ymin, self.region_b.ymin) - self.field_padding
        ymax = max(self.region_a.ymax, self.region_b.ymax) + self.field_padding

        # X bounds
        if s["pball"][0] - self.ball_radius < xmin:
            s["pball"][0] = xmin + self.ball_radius
            s["vball"][0] = -s["vball"][0] * self.restitution
        elif s["pball"][0] + self.ball_radius > xmax:
            s["pball"][0] = xmax - self.ball_radius
            s["vball"][0] = -s["vball"][0] * self.restitution

        # Y bounds
        if s["pball"][1] - self.ball_radius < ymin:
            s["pball"][1] = ymin + self.ball_radius
            s["vball"][1] = -s["vball"][1] * self.restitution
        elif s["pball"][1] + self.ball_radius > ymax:
            # NEW/CHANGED: goal mouth is an opening; don't bounce there
            in_mouth = (self.goal_xmin <= s["pball"][0] <= self.goal_xmax)
            if not in_mouth:
                s["pball"][1] = ymax - self.ball_radius
                s["vball"][1] = -s["vball"][1] * self.restitution
            # else: allow pass-through (no clamp, no reflection)

        # enforce max speed for ball after impulses and bounces
        s["vball"] = cap(s["vball"], self.max_speed_ball)

        # Termination & reward
        terminated = False
        reward = 0.01
        left_regions = False

        if not self.region_a.contains_point(s["pA"]) or not self.region_b.contains_point(s["pB"]):
            reward = -10.0
            terminated = True
            left_regions = True

        # Detect crossing the goal line inside the posts
        crossed_goal_line = (
            s["pball"][1] >= self.region_b.ymax and
            self.goal_xmin <= s["pball"][0] <= self.goal_xmax and
            self.region_b.xmin <= s["pball"][0] <= self.region_b.xmax
        )
        # Only count as a goal if last touch was by B (if you have that rule)
        is_goal = bool(crossed_goal_line and (s.get("last_touch", "B") == "B"))  # adapt if you removed last_touch

        if is_goal:
            reward = 10.0
            terminated = True

        s["steps"] += 1
        truncated = (s["steps"] >= self.max_steps) and not terminated

        obs = self._get_obs()
        info_after = self._get_info()
        info_after["crossed_goal_mouth"] = bool(crossed_goal_line)
        info_after["scored_goal"] = bool(is_goal)
        info_after["left_regions"] = bool(left_regions)
        info["after"] = info_after
        return obs, reward, terminated, truncated, info

    def _dist_to_goal_mouth(self, bx: float, by: float) -> float:
        # vertical gap (only if below the line)
        dy = max(0.0, self.goal_y - float(by))

        # horizontal gap to the segment (0 if within posts)
        if self.goal_xmin <= float(bx) <= self.goal_xmax:
            dx = 0.0
        else:
            dx = min(abs(float(bx) - self.goal_xmin), abs(float(bx) - self.goal_xmax))

        # Euclidean distance in the forward half-space
        return float(math.hypot(dx, dy))

    def _relative(self, ref_p: np.ndarray, ref_v: np.ndarray, obj_p: np.ndarray, obj_v: np.ndarray):
        # no orientation; relative = simple difference in world frame
        rel_p = obj_p - ref_p
        rel_v = obj_v - ref_v
        return rel_p, rel_v

    def _get_obs(self):
        s = self.state
        if self.centralized:
            return np.array([
                *s["pA"], *s["pB"], *s["vA"], *s["vB"], *s["pball"], *s["vball"]
            ], dtype=np.float32)
        else:
            p_rel_ball_A, v_rel_ball_A = self._relative(s["pA"], s["vA"], s["pball"], s["vball"])
            p_rel_B_from_A, v_rel_B_from_A = self._relative(s["pA"], s["vA"], s["pB"], s["vB"])

            p_rel_ball_B, v_rel_ball_B = self._relative(s["pB"], s["vB"], s["pball"], s["vball"])
            p_rel_A_from_B, v_rel_A_from_B = self._relative(s["pB"], s["vB"], s["pA"], s["vA"])

            oA = np.array([*s["pA"], *s["vA"],
                           *p_rel_ball_A, *v_rel_ball_A,
                           *p_rel_B_from_A, *v_rel_B_from_A], dtype=np.float32)
            oB = np.array([*s["pB"], *s["vB"],
                           *p_rel_ball_B, *v_rel_ball_B,
                           *p_rel_A_from_B, *v_rel_A_from_B], dtype=np.float32)
            return (oA, oB)
    
    def _get_info(self):
        s = self.state
        dist_B_to_ball = float(np.linalg.norm(s["pB"] - s["pball"]))
        info = {
            "ball_x": float(s["pball"][0]),
            "ball_y": float(s["pball"][1]),
            "dist_ball_to_goal": self._dist_to_goal_mouth(s["pball"][0], s["pball"][1]),
            "dist_A_to_ball": float(np.linalg.norm(s["pA"] - s["pball"])),
            "dist_B_to_ball": dist_B_to_ball,
            "passed_regions": bool(self.region_b.contains_point(s["pball"])),  # ball in B?
            "speed_ball": float(np.linalg.norm(s["vball"])),
            "last_touch": s.get("last_touch", "A"),
        }
        return info

    def render(self):
        s = self.state
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(6.5, 4))
        ax = self.ax
        ax.clear()

        xmin = self.region_a.xmin - self.field_padding
        xmax = self.region_b.xmax + self.field_padding
        ymin = min(self.region_a.ymin, self.region_b.ymin) - self.field_padding
        ymax = max(self.region_a.ymax, self.region_b.ymax) + self.field_padding

        ax.add_patch(plt.Rectangle((self.region_a.xmin, self.region_a.ymin),
                       self.region_a.xmax - self.region_a.xmin,
                       self.region_a.ymax - self.region_a.ymin,
                       fill=False, linewidth=2))
        ax.text((self.region_a.xmin+self.region_a.xmax)/2, self.region_a.ymax+0.05, "Region A", ha="center", va="bottom")

        ax.add_patch(plt.Rectangle((self.region_b.xmin, self.region_b.ymin),
                       self.region_b.xmax - self.region_b.xmin,
                       self.region_b.ymax - self.region_b.ymin,
                       fill=False, linewidth=2))
        ax.text((self.region_b.xmin+self.region_b.xmax)/2, self.region_b.ymax+0.05, "Region B", ha="center", va="bottom")

        ax.plot([self.goal_xmin, self.goal_xmax], [self.region_b.ymax, self.region_b.ymax], linewidth=6)

        def draw_agent(p, v, label):
            circ = plt.Circle((p[0], p[1]), self.agent_radius, fill=False, linewidth=2)
            ax.add_patch(circ)
            # arrow direction from velocity if moving, else upward
            speed = np.linalg.norm(v)
            if speed > 1e-6:
                head = (v / (speed + 1e-12)) * (self.agent_radius * 1.5)
            else:
                head = np.array([0.0, 1.0]) * (self.agent_radius * 1.5)
            ax.arrow(p[0], p[1], head[0], head[1], head_width=0.02, length_includes_head=True)
            ax.text(p[0], p[1]-0.08, label, ha="center", va="top")

        draw_agent(s["pA"], s["vA"], "A")
        draw_agent(s["pB"], s["vB"], "B")

        ball = plt.Circle((s["pball"][0], s["pball"][1]), self.ball_radius, fill=True)
        ax.add_patch(ball)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Pass-and-Score: Episode State")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Build a compact, human-readable representation of the state
        def fmt_item(x):
            try:
                it = list(x)
                return "[" + ", ".join(f"{float(v):.2f}" for v in it) + "]"
            except Exception:
                return f"{float(x):.2f}"

        # NEW/CHANGED: include last_touch in the HUD text
        text_lines = [
            f"steps: {int(s.get('steps', 0))}",
            f"last_touch: {s.get('last_touch', 'A')}",
            f"pA: {fmt_item(s['pA'])}",
            f"vA: {fmt_item(s['vA'])}",
            f"pB: {fmt_item(s['pB'])}",
            f"vB: {fmt_item(s['vB'])}",
            f"pball: {fmt_item(s['pball'])}",
            f"vball: {fmt_item(s['vball'])}",
        ]
        state_text = "\n".join(text_lines)

        ax.text(0.98, 0.98, state_text, transform=ax.transAxes,
            fontsize=8, ha="right", va="top",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"))

        plt.show()

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
        self.fig, self.ax = None, None
