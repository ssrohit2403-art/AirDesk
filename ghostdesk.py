# ghostdesk.py
"""
GhostDesk - single-file touchless control prototype.
Python 3.12 | OpenCV | MediaPipe 0.10.21 | PyAutoGUI | NumPy | Windows
"""

import time
import math

import cv2
import numpy as np
import mediapipe as mp
import pyautogui

# ============================================================
# SETTINGS - tune these
# ============================================================
CAMERA_INDEX = 0
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540

CURSOR_SMOOTHING = 0.35        # 0..1, lower = smoother/laggier, higher = snappier/jittery
CURSOR_SENSITIVITY = 1.3       # >1 amplifies hand movement into cursor movement
CURSOR_DEAD_ZONE_PX = 4        # ignore cursor moves smaller than this many pixels

PINCH_THRESHOLD = 0.45         # thumb-index distance (normalized) below this = pinched
PINCH_RELEASE_THRESHOLD = 0.55 # must open past this before another click can fire
RIGHT_CLICK_THRESHOLD = 0.45   # thumb-middle distance (normalized) below this = pinched
RIGHT_CLICK_RELEASE_THRESHOLD = 0.55
RIGHT_CLICK_COOLDOWN = 0.35    # seconds between right clicks

SCROLL_THRESHOLD = 0.01        # min normalized vertical movement per frame to scroll
SCROLL_SPEED = 900             # multiplier: hand movement -> scroll amount

LOCK_HOLD_TIME = 0.35          # seconds a fist must be held to lock
UNLOCK_HOLD_TIME = 0.5         # seconds an open hand must be held to unlock

DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.6

# Interaction region inside the camera frame that maps to the FULL screen.
# (x0, y0, x1, y1) in normalized 0..1 camera coordinates.
REGION_MARGIN = 0.15
ACTIVE_REGION = (REGION_MARGIN, REGION_MARGIN, 1 - REGION_MARGIN, 1 - REGION_MARGIN)

MIRROR_CAMERA = True
SHOW_HUD = True

ACCENT = (255, 210, 0)     # cyan-ish (BGR)
WARN = (0, 90, 255)        # orange/red (BGR)
TEXT_COLOR = (230, 230, 230)
FONT = cv2.FONT_HERSHEY_SIMPLEX

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

# MediaPipe landmark indices used
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_TIP = 5, 8
MIDDLE_MCP, MIDDLE_TIP = 9, 12
RING_MCP, RING_TIP = 13, 16
PINKY_MCP, PINKY_TIP = 17, 20


# ============================================================
# GEOMETRY HELPERS
# ============================================================
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def hand_scale(points):
    """Wrist-to-middle-knuckle distance. Used to normalize pinch distances
    so gestures work the same whether the hand is close to or far from the
    camera."""
    return max(dist(points[WRIST], points[MIDDLE_MCP]), 1e-4)


def finger_extended(points, tip_idx, pip_idx):
    """A finger counts as extended if its tip is meaningfully further from
    the wrist than its own pip knuckle is. Works across hand rotation
    better than a plain vertical y-comparison."""
    tip_to_wrist = dist(points[tip_idx], points[WRIST])
    pip_to_wrist = dist(points[pip_idx], points[WRIST])
    return tip_to_wrist > pip_to_wrist * 1.1


def curled_finger_count(points):
    """Counts how many of the four main fingers (index, middle, ring,
    pinky) are curled (NOT extended). Used for fist detection."""
    fingers = [(INDEX_TIP, 6), (MIDDLE_TIP, 10), (RING_TIP, 14), (PINKY_TIP, 18)]
    curled = 0
    for tip, pip in fingers:
        if not finger_extended(points, tip, pip):
            curled += 1
    return curled


def is_fist(points):
    # Require at least 3 curled fingers (not just one) so a fist can't be
    # triggered by accident while doing something else with one finger.
    return curled_finger_count(points) >= 3


def is_open_hand(points):
    fingers = [(INDEX_TIP, 6), (MIDDLE_TIP, 10), (RING_TIP, 14), (PINKY_TIP, 18)]
    extended = sum(1 for tip, pip in fingers if finger_extended(points, tip, pip))
    return extended >= 4


def is_scroll_pose(points):
    """Index + middle raised, ring + pinky lowered."""
    index_up = finger_extended(points, INDEX_TIP, 6)
    middle_up = finger_extended(points, MIDDLE_TIP, 10)
    ring_up = finger_extended(points, RING_TIP, 14)
    pinky_up = finger_extended(points, PINKY_TIP, 18)
    return index_up and middle_up and not ring_up and not pinky_up


def palm_center(points):
    xs = [points[WRIST][0], points[INDEX_MCP][0], points[PINKY_MCP][0]]
    ys = [points[WRIST][1], points[INDEX_MCP][1], points[PINKY_MCP][1]]
    return (sum(xs) / 3.0, sum(ys) / 3.0)


def pinch_distance(points, tip_idx):
    return dist(points[tip_idx], points[THUMB_TIP]) / hand_scale(points)


# ============================================================
# ACTION CONTROLLER - the only place that touches pyautogui
# ============================================================
class ActionController:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()

    def move_mouse(self, x, y):
        x = max(0, min(self.screen_w - 1, int(x)))
        y = max(0, min(self.screen_h - 1, int(y)))
        pyautogui.moveTo(x, y)

    def left_click(self):
        pyautogui.click()

    def right_click(self):
        pyautogui.rightClick()

    def scroll(self, amount):
        pyautogui.scroll(amount)

    def release_all(self):
        # Belt-and-braces: this build never holds a modifier key down, but
        # this keeps the shutdown path future-proof and cheap to call.
        for key in ("alt", "shift", "ctrl"):
            try:
                pyautogui.keyUp(key)
            except Exception:
                pass


class EMAFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.value = None

    def update(self, sample):
        if self.value is None:
            self.value = sample
        else:
            self.value = self.alpha * sample + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


# ============================================================
# GESTURE STATE MACHINE
# ============================================================
class GestureState:
    def __init__(self, actions: ActionController):
        self.actions = actions

        self.cursor_x_filter = EMAFilter(CURSOR_SMOOTHING)
        self.cursor_y_filter = EMAFilter(CURSOR_SMOOTHING)
        self.last_cursor_pos = None

        self.index_pinched = False
        self.middle_pinched = False
        self.last_right_click_time = 0.0

        self.scroll_prev_y = None

        self.fist_start_time = None
        self.open_start_time = None
        self.locked = False

        self.current_gesture = "NONE"
        self.current_action = "NONE"
        self.cursor_screen_pos = (0, 0)

    def reset_transient(self):
        self.cursor_x_filter.reset()
        self.cursor_y_filter.reset()
        self.scroll_prev_y = None
        self.current_gesture = "NONE"
        self.current_action = "NONE"

    def process_no_hand(self):
        self.reset_transient()
        self.fist_start_time = None
        self.open_start_time = None

    def process(self, points, now):
        # ---- LOCK always has top priority ----
        if is_fist(points):
            self.open_start_time = None
            if self.fist_start_time is None:
                self.fist_start_time = now
            elif not self.locked and (now - self.fist_start_time) >= LOCK_HOLD_TIME:
                self.locked = True
                self.current_gesture = "FIST"
                self.current_action = "LOCKED"
        else:
            self.fist_start_time = None

        if self.locked:
            if is_open_hand(points):
                if self.open_start_time is None:
                    self.open_start_time = now
                elif (now - self.open_start_time) >= UNLOCK_HOLD_TIME:
                    self.locked = False
                    self.open_start_time = None
                    self.current_gesture = "OPEN HAND"
                    self.current_action = "UNLOCKED"
            else:
                self.open_start_time = None
            self.current_gesture = self.current_gesture if self.locked else self.current_gesture
            self.current_action = "LOCKED" if self.locked else self.current_action
            return  # nothing else runs while locked

        # ---- SCROLL MODE ----
        if is_scroll_pose(points):
            self.current_gesture = "SCROLL POSE"
            _, y = palm_center(points)
            if self.scroll_prev_y is None:
                self.scroll_prev_y = y
            else:
                delta = self.scroll_prev_y - y  # positive = hand moved up
                self.scroll_prev_y = y
                if abs(delta) >= SCROLL_THRESHOLD:
                    amount = int(delta * SCROLL_SPEED)
                    if amount != 0:
                        self.actions.scroll(amount)
                        self.current_action = f"SCROLL {'UP' if amount > 0 else 'DOWN'}"
            return
        else:
            self.scroll_prev_y = None

        # ---- CLICK GESTURES ----
        self._process_clicks(points, now)

        # ---- POINTER (default) ----
        self._process_pointer(points)

    def _process_clicks(self, points, now):
        left_dist = pinch_distance(points, INDEX_TIP)
        right_dist = pinch_distance(points, MIDDLE_TIP)

        # Left click: thumb + index
        if not self.index_pinched and left_dist < PINCH_THRESHOLD:
            self.index_pinched = True
            self.actions.left_click()
            self.current_gesture = "PINCH (INDEX)"
            self.current_action = "LEFT CLICK"
        elif self.index_pinched and left_dist > PINCH_RELEASE_THRESHOLD:
            self.index_pinched = False  # wait-for-release satisfied

        # Right click: thumb + middle (only if not already left-pinching)
        if not self.index_pinched:
            if (not self.middle_pinched and right_dist < RIGHT_CLICK_THRESHOLD
                    and (now - self.last_right_click_time) > RIGHT_CLICK_COOLDOWN):
                self.middle_pinched = True
                self.actions.right_click()
                self.last_right_click_time = now
                self.current_gesture = "PINCH (MIDDLE)"
                self.current_action = "RIGHT CLICK"
            elif self.middle_pinched and right_dist > RIGHT_CLICK_RELEASE_THRESHOLD:
                self.middle_pinched = False

    def _process_pointer(self, points):
        x0, y0, x1, y1 = ACTIVE_REGION
        fx, fy = points[INDEX_TIP]

        norm_x = (fx - x0) / max(1e-6, (x1 - x0))
        norm_y = (fy - y0) / max(1e-6, (y1 - y0))
        norm_x = min(1.0, max(0.0, norm_x))
        norm_y = min(1.0, max(0.0, norm_y))

        # Amplify around center for sensitivity
        norm_x = 0.5 + (norm_x - 0.5) * CURSOR_SENSITIVITY
        norm_y = 0.5 + (norm_y - 0.5) * CURSOR_SENSITIVITY
        norm_x = min(1.0, max(0.0, norm_x))
        norm_y = min(1.0, max(0.0, norm_y))

        target_x = norm_x * self.actions.screen_w
        target_y = norm_y * self.actions.screen_h

        smooth_x = self.cursor_x_filter.update(target_x)
        smooth_y = self.cursor_y_filter.update(target_y)

        if self.last_cursor_pos is not None:
            dx = smooth_x - self.last_cursor_pos[0]
            dy = smooth_y - self.last_cursor_pos[1]
            if math.hypot(dx, dy) < CURSOR_DEAD_ZONE_PX:
                smooth_x, smooth_y = self.last_cursor_pos

        self.actions.move_mouse(smooth_x, smooth_y)
        self.last_cursor_pos = (smooth_x, smooth_y)
        self.cursor_screen_pos = (int(smooth_x), int(smooth_y))

        if not self.index_pinched and not self.middle_pinched:
            self.current_gesture = "POINTING"
            self.current_action = "MOVE CURSOR"


# ============================================================
# HUD DRAWING
# ============================================================
def draw_active_region(frame, w, h):
    x0, y0, x1, y1 = ACTIVE_REGION
    p0 = (int(x0 * w), int(y0 * h))
    p1 = (int(x1 * w), int(y1 * h))
    cv2.rectangle(frame, p0, p1, (90, 90, 90), 1, cv2.LINE_AA)


def draw_landmarks(frame, points, w, h):
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    ]
    for a, b in connections:
        pa = (int(points[a][0] * w), int(points[a][1] * h))
        pb = (int(points[b][0] * w), int(points[b][1] * h))
        cv2.line(frame, pa, pb, ACCENT, 1, cv2.LINE_AA)
    for p in points:
        px = (int(p[0] * w), int(p[1] * h))
        cv2.circle(frame, px, 2, ACCENT, -1, cv2.LINE_AA)

    index_px = (int(points[INDEX_TIP][0] * w), int(points[INDEX_TIP][1] * h))
    thumb_px = (int(points[THUMB_TIP][0] * w), int(points[THUMB_TIP][1] * h))
    cv2.circle(frame, index_px, 8, (0, 255, 180), 2, cv2.LINE_AA)
    cv2.circle(frame, thumb_px, 6, (255, 160, 0), 2, cv2.LINE_AA)


def panel(frame, x, y, w, h, alpha=0.4, color=(20, 20, 20)):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_hud(frame, w, h, fps, tracking_ok, gesture, action, locked, cursor_pos):
    panel(frame, 0, 0, w, 44, alpha=0.45)
    title_color = WARN if locked else ACCENT
    cv2.putText(frame, "GHOSTDESK // ONLINE", (12, 28), FONT, 0.65,
                title_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS {fps:4.1f}", (w - 130, 28), FONT, 0.6,
                TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.line(frame, (0, 44), (w, 44), title_color, 1, cv2.LINE_AA)

    status_lines = [
        f"STATUS   : {'LOCKED' if locked else 'ACTIVE'}",
        f"TRACKING : {'OK' if tracking_ok else 'LOST'}",
        f"GESTURE  : {gesture}",
        f"ACTION   : {action}",
        f"CURSOR   : {cursor_pos[0]}, {cursor_pos[1]}",
    ]
    panel_h = 24 * len(status_lines) + 14
    panel(frame, 8, h - panel_h - 8, 300, panel_h, alpha=0.4)
    y = h - panel_h + 10
    for line in status_lines:
        color = WARN if ("LOCKED" in line and locked) else TEXT_COLOR
        cv2.putText(frame, line, (18, y), FONT, 0.48, color, 1, cv2.LINE_AA)
        y += 24

    cv2.putText(frame, "C:none  H:Toggle HUD  Q:Quit", (12, h - 12), FONT,
                0.45, (150, 150, 150), 1, cv2.LINE_AA)


def draw_tracking_lost(frame, w, h):
    panel(frame, w // 2 - 150, h // 2 - 28, 300, 56, alpha=0.55, color=(0, 0, 0))
    cv2.putText(frame, "TRACKING LOST", (w // 2 - 120, h // 2 + 8), FONT,
                0.85, WARN, 2, cv2.LINE_AA)


def draw_locked_banner(frame, w, h):
    panel(frame, 0, h // 2 - 36, w, 72, alpha=0.55, color=(0, 0, 0))
    text = "GHOSTDESK LOCKED"
    size = cv2.getTextSize(text, FONT, 1.0, 2)[0]
    cv2.putText(frame, text, ((w - size[0]) // 2, h // 2 + 8), FONT, 1.0,
                WARN, 2, cv2.LINE_AA)
    sub = "Open hand and hold to unlock"
    sub_size = cv2.getTextSize(sub, FONT, 0.5, 1)[0]
    cv2.putText(frame, sub, ((w - sub_size[0]) // 2, h // 2 + 32), FONT, 0.5,
                TEXT_COLOR, 1, cv2.LINE_AA)


# ============================================================
# MAIN LOOP
# ============================================================
def main():
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("ERROR: could not open webcam. Check CAMERA_INDEX.")
        return

    actions = ActionController()
    state = GestureState(actions)

    show_hud = SHOW_HUD
    fps = 0.0
    last_time = time.time()
    tracking_lost_frames = 0
    TRACKING_LOST_GRACE = 3

    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    ) as hands:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("WARNING: failed to read frame")
                    continue

                if MIRROR_CAMERA:
                    frame = cv2.flip(frame, 1)

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = hands.process(rgb)

                now = time.time()
                tracking_ok = False
                points = None

                if results.multi_hand_landmarks:
                    lm = results.multi_hand_landmarks[0]
                    points = [(p.x, p.y) for p in lm.landmark]
                    tracking_ok = True
                    tracking_lost_frames = 0
                    state.process(points, now)
                else:
                    tracking_lost_frames += 1
                    if tracking_lost_frames > TRACKING_LOST_GRACE:
                        state.process_no_hand()

                # ---- FPS ----
                dt = max(1e-6, now - last_time)
                last_time = now
                instant_fps = 1.0 / dt
                fps = 0.9 * fps + 0.1 * instant_fps

                # ---- DRAW ----
                if show_hud:
                    draw_active_region(frame, w, h)
                    if tracking_ok:
                        draw_landmarks(frame, points, w, h)
                    draw_hud(frame, w, h, fps,
                             tracking_ok=(tracking_lost_frames <= TRACKING_LOST_GRACE),
                             gesture=state.current_gesture,
                             action=state.current_action,
                             locked=state.locked,
                             cursor_pos=state.cursor_screen_pos)
                    if not tracking_ok and tracking_lost_frames > TRACKING_LOST_GRACE:
                        draw_tracking_lost(frame, w, h)
                    if state.locked:
                        draw_locked_banner(frame, w, h)

                cv2.imshow("GhostDesk // Vision Interface", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('h'):
                    show_hud = not show_hud

        finally:
            # Guaranteed cleanup: no stuck modifier keys, camera released,
            # windows closed, no matter how the loop above exits.
            actions.release_all()
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()