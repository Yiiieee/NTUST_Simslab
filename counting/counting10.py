import json
import time
import os
import cv2
import requests
from ultralytics import YOLO
from urllib.parse import urlparse, parse_qs

# pip install pyzbar
from pyzbar.pyzbar import decode as zbar_decode

# ===================== CONFIG =====================
MODEL_PATH = "runs/detect/runs/model5_yolov26n/weights/best.pt"
# VIDEO_PATH = "data-original/2-4.mp4"
VIDEO_PATH = "0"
CONF_THRESHOLD = 0.65

QR_READ_EVERY_N_FRAMES = 5
MIN_UPDATE_INTERVAL = 0.2  # network throttle

SUPABASE_URL = "https://yrsdedmvnpavswfjfmdy.supabase.co"
SUPABASE_KEY = os.getenv(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlyc2RlZG12bnBhdnN3ZmpmbWR5Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2OTE2MzI1NSwiZXhwIjoyMDg0NzM5MjU1fQ.d_EatbYxL22Tr4eTqXagf8LdF7IqWX82EUOtXfnOsYo"
).strip()

TABLE = "Boxes"
ID_FIELD = "id"
STOCK_FIELD = "Stock"

STATE_FILE = "state.json"

# QR ROI (y1, y2, x1, x2)
QR_ROI = (100, 650, 100, 450)

SHOW_QR_DEBUG = True
SHOW_MAIN_WINDOW = True
SHOW_GHOST_BOX = True

# ===================== 2 BOX LOGIC =====================
MAIN_BOX = (550, 420, 1000, 650)
HUMAN_BOX = (550, 100, 1000, 420)

ZONE_MARGIN = 12
ZONE_CONFIRM_FRAMES = 3

# ===================== GHOST MATCH CONFIG =====================
GHOST_MAX_MISSING_SECONDS = 3.0
GHOST_MATCH_RADIUS = 150
GLOBAL_TTL_SECONDS = 8.0
TRACK_TTL_SECONDS = 4.0

# ===================== CANDIDATE TRANSFER CONFIG =====================
TRANSFER_WINDOW_SECONDS = 4.0
MIN_STABLE_FRAMES_BEFORE_TRANSFER = 3

# Radius for synchronizing taken state across GIDs
# that may represent the same physical object
TAKEN_SYNC_RADIUS = 180


# ===================== SUPABASE REST =====================
class SupabaseREST:
    def __init__(self, base_url: str, key: str):
        self.base_url = base_url.rstrip("/")
        self.key = (key or "").strip()

    def headers(self):
        headers = {"Accept": "application/json"}
        if self.key:
            headers["apikey"] = self.key
            headers["Authorization"] = f"Bearer {self.key}"
        return headers

    def fetch_one(self, box_id: int):
        url = f"{self.base_url}/rest/v1/{TABLE}?{ID_FIELD}=eq.{box_id}&select=*"
        response = requests.get(url, headers=self.headers(), timeout=8)
        response.raise_for_status()
        data = response.json()
        if not data:
            raise ValueError(f"ID {box_id} not found")
        return data[0]

    def update_stock(self, box_id: int, new_stock: int):
        if not self.key or self.key == "YOUR_SUPABASE_KEY_HERE":
            raise ValueError("SUPABASE_KEY is empty. Set the SUPABASE_KEY environment variable first.")

        url = f"{self.base_url}/rest/v1/{TABLE}?{ID_FIELD}=eq.{box_id}"
        payload = {STOCK_FIELD: int(new_stock)}

        headers = self.headers()
        headers["Content-Type"] = "application/json"
        headers["Prefer"] = "return=representation"

        response = requests.patch(url, headers=headers, json=payload, timeout=8)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list) or len(data) == 0:
            return self.fetch_one(box_id)
        return data[0]


# ===================== QR PARSER =====================
def parse_box_id_from_qr(text: str) -> int:
    text = (text or "").strip()
    if not text:
        raise ValueError("QR is empty")

    if text.startswith("{") and text.endswith("}"):
        obj = json.loads(text)
        if "id" not in obj:
            raise ValueError("QR JSON does not contain key 'id'")
        return int(obj["id"])

    if text.isdigit():
        return int(text)

    parsed = urlparse(text)
    if parsed.scheme in ("http", "https"):
        query_params = parse_qs(parsed.query)
        if "id" in query_params and query_params["id"]:
            value = query_params["id"][0]
            if isinstance(value, str) and value.startswith("eq."):
                value = value[3:]
            if str(value).isdigit():
                return int(value)

    raise ValueError("Unknown QR format")


# ===================== STATE WRITER =====================
def write_state(active_box_id=None, last_event="Ready", last_decrement=None):
    temp_file = STATE_FILE + ".tmp"
    payload = {
        "ts": time.time(),
        "active_box_id": active_box_id,
        "last_event": last_event,
        "last_decrement": last_decrement,
    }
    with open(temp_file, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(temp_file, STATE_FILE)


# ===================== QR HELPERS =====================
def preprocess_for_qr(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray, threshold


def decode_qr_pyzbar(image_bgr):
    decoded = zbar_decode(image_bgr)
    if decoded:
        return decoded[0].data.decode("utf-8", errors="ignore")

    _, threshold = preprocess_for_qr(image_bgr)
    decoded = zbar_decode(threshold)
    if decoded:
        return decoded[0].data.decode("utf-8", errors="ignore")

    return None


# ===================== ZONE HELPERS =====================
def point_in_box(px, py, box):
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def point_in_inner_box(px, py, box, margin=0):
    x1, y1, x2, y2 = box
    return (x1 + margin) <= px <= (x2 - margin) and (y1 + margin) <= py <= (y2 - margin)


def get_zone(px, py, main_box, human_box, margin=10):
    in_main_outer = point_in_box(px, py, main_box)
    in_human_outer = point_in_box(px, py, human_box)

    in_main_inner = point_in_inner_box(px, py, main_box, margin=margin)
    in_human_inner = point_in_inner_box(px, py, human_box, margin=margin)

    if in_main_inner and not in_human_outer:
        return "main"

    if in_human_inner and not in_main_outer:
        return "human"

    if (not in_main_outer) and (not in_human_outer):
        return "outside"

    return "ambiguous"


def update_zone_stability(track_state, observed_zone):
    previous_stable_zone = track_state["last_stable_zone"]

    if observed_zone == "ambiguous":
        return False, previous_stable_zone, previous_stable_zone

    if observed_zone == track_state["candidate_zone"]:
        track_state["candidate_count"] += 1
    else:
        track_state["candidate_zone"] = observed_zone
        track_state["candidate_count"] = 1

    if (
        observed_zone != track_state["last_stable_zone"]
        and track_state["candidate_count"] >= ZONE_CONFIRM_FRAMES
    ):
        track_state["last_stable_zone"] = observed_zone
        return True, previous_stable_zone, observed_zone

    return False, previous_stable_zone, track_state["last_stable_zone"]


# ===================== GEOMETRY HELPERS =====================
def xywh_to_xyxy(x, y, w, h):
    x1 = float(x - w / 2)
    y1 = float(y - h / 2)
    x2 = float(x + w / 2)
    y2 = float(y + h / 2)
    return x1, y1, x2, y2


def euclidean_dist(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def bottom_center_from_xywh(x, y, w, h):
    return float(x), float(y + h / 2.0)


def to_int_box(box):
    x1, y1, x2, y2 = box
    return int(x1), int(y1), int(x2), int(y2)


def safe_int_frames(seconds, fps, minimum=1):
    return max(minimum, int(round(seconds * fps)))


# ===================== GLOBAL STATE =====================
def init_global_state(frame_idx, initial_zone="outside"):
    return {
        "last_stable_zone": initial_zone,
        "candidate_zone": initial_zone,
        "candidate_count": 1,

        "taken": False,
        "counted": False,

        "last_seen_frame": frame_idx,
        "last_bottom_center": None,
        "last_bbox": None,

        "current_tracker_id": None,
        "tracker_ids": [],

        "is_ghost": False,
        "ghost_since_frame": None,

        # Candidate transfer
        "zone_stable_frames": 0,
        "transfer_candidate": False,
        "transfer_from_zone": None,
        "transfer_open_frame": None,
        "transfer_anchor_point": None,
    }


# ===================== STATE SYNC =====================
def sync_taken_state_related_gids(global_states, base_gid, new_taken):
    """
    Synchronize the 'taken' flag to other GIDs that most likely
    represent the same physical object.
    Practical patch because state is still stored per GID.
    """
    if base_gid not in global_states:
        return

    base_state = global_states[base_gid]
    base_point = base_state.get("last_bottom_center", None)

    # Update base itself
    base_state["taken"] = new_taken

    if base_point is None:
        return

    for gid, state in global_states.items():
        if gid == base_gid:
            continue

        other_point = state.get("last_bottom_center", None)
        if other_point is None:
            continue

        dist = euclidean_dist(base_point, other_point)
        if dist <= TAKEN_SYNC_RADIUS:
            state["taken"] = new_taken


# ===================== GHOST MATCH =====================
def find_matching_ghost(
    det_bottom_center,
    global_states,
    active_global_ids_this_frame,
    current_frame_idx,
    ghost_max_gap_frames,
    ghost_match_radius
):
    best_gid = None
    best_dist = 1e18

    for gid, state in global_states.items():
        if gid in active_global_ids_this_frame:
            continue

        if not state.get("is_ghost", False):
            continue

        if state.get("last_bottom_center") is None:
            continue

        gap = current_frame_idx - state["last_seen_frame"]
        if gap <= 0 or gap > ghost_max_gap_frames:
            continue

        dist = euclidean_dist(det_bottom_center, state["last_bottom_center"])
        if dist <= ghost_match_radius and dist < best_dist:
            best_dist = dist
            best_gid = gid

    return best_gid, best_dist


def assign_global_id_simple(
    tid,
    det_bottom_center,
    tracker_to_global,
    global_states,
    active_global_ids_this_frame,
    next_global_id_ref,
    current_frame_idx,
    ghost_max_gap_frames,
    ghost_match_radius
):
    if tid in tracker_to_global:
        gid = tracker_to_global[tid]
        active_global_ids_this_frame.add(gid)
        return gid, "direct", None

    matched_gid, matched_dist = find_matching_ghost(
        det_bottom_center=det_bottom_center,
        global_states=global_states,
        active_global_ids_this_frame=active_global_ids_this_frame,
        current_frame_idx=current_frame_idx,
        ghost_max_gap_frames=ghost_max_gap_frames,
        ghost_match_radius=ghost_match_radius
    )

    if matched_gid is not None:
        tracker_to_global[tid] = matched_gid
        active_global_ids_this_frame.add(matched_gid)
        return matched_gid, "ghost_match", matched_dist

    gid = next_global_id_ref[0]
    next_global_id_ref[0] += 1
    tracker_to_global[tid] = gid
    active_global_ids_this_frame.add(gid)
    return gid, "new", None


def cleanup_old_tracker_mappings(tracker_to_global, tracker_last_seen, current_frame_idx, ttl_frames=60):
    expired_tids = []
    for tid, last_seen in tracker_last_seen.items():
        if current_frame_idx - last_seen > ttl_frames:
            expired_tids.append(tid)

    for tid in expired_tids:
        tracker_last_seen.pop(tid, None)
        tracker_to_global.pop(tid, None)


def cleanup_old_globals(global_states, current_frame_idx, ttl_frames=120):
    expired_gids = []
    for gid, info in global_states.items():
        if info["last_seen_frame"] is None:
            expired_gids.append(gid)
            continue

        if current_frame_idx - info["last_seen_frame"] > ttl_frames:
            expired_gids.append(gid)

    for gid in expired_gids:
        global_states.pop(gid, None)


# ===================== DB EVENT =====================
def apply_stock_change(supa, active_box_id, delta, event_name, last_update_time):
    if not active_box_id:
        return last_update_time, False, f"{event_name} but NO QR ID"

    now = time.time()
    if now - last_update_time < MIN_UPDATE_INTERVAL:
        return last_update_time, False, f"{event_name} skipped بسبب throttle"

    try:
        row = supa.fetch_one(active_box_id)
        before = int(row.get(STOCK_FIELD, 0))
        after = max(0, before + delta) if delta < 0 else before + delta
        supa.update_stock(active_box_id, after)

        write_state(
            active_box_id=active_box_id,
            last_event=f"{event_name} | {active_box_id}: {before}->{after}",
            last_decrement={
                "id": active_box_id,
                "before": before,
                "after": after,
                "delta": delta,
                "event": event_name
            }
        )
        return now, True, f"{event_name} | {active_box_id}: {before}->{after}"

    except Exception as e:
        write_state(
            active_box_id=active_box_id,
            last_event=f"DB ERROR ({event_name}): {str(e)[:120]}",
            last_decrement=None
        )
        return last_update_time, False, f"DB ERROR ({event_name}): {str(e)}"


# ===================== CANDIDATE TRANSFER =====================
def open_transfer_candidate(state, frame_idx):
    last_zone = state.get("last_stable_zone", "outside")
    if last_zone not in ("main", "human"):
        return False

    if state.get("zone_stable_frames", 0) < MIN_STABLE_FRAMES_BEFORE_TRANSFER:
        return False

    state["transfer_candidate"] = True
    state["transfer_from_zone"] = last_zone
    state["transfer_open_frame"] = frame_idx
    state["transfer_anchor_point"] = state.get("last_bottom_center")
    return True


def close_transfer_candidate(state):
    state["transfer_candidate"] = False
    state["transfer_from_zone"] = None
    state["transfer_open_frame"] = None
    state["transfer_anchor_point"] = None


def candidate_is_alive(state, frame_idx, transfer_window_frames):
    if not state.get("transfer_candidate", False):
        return False
    open_frame = state.get("transfer_open_frame", None)
    if open_frame is None:
        return False
    return (frame_idx - open_frame) <= transfer_window_frames


def expected_target_zone(from_zone):
    if from_zone == "main":
        return "human"
    if from_zone == "human":
        return "main"
    return None


def resolve_same_gid_transfer(
    global_states,
    frame_idx,
    transfer_window_frames,
    supa,
    active_box_id,
    last_update_time,
):
    messages = []
    take_delta = 0
    return_delta = 0

    for gid, state in global_states.items():
        if not candidate_is_alive(state, frame_idx, transfer_window_frames):
            continue

        current_zone = state.get("last_stable_zone")
        from_zone = state.get("transfer_from_zone")

        if current_zone not in ("main", "human"):
            continue

        if state.get("is_ghost", False):
            continue

        if state.get("zone_stable_frames", 0) < MIN_STABLE_FRAMES_BEFORE_TRANSFER:
            continue

        # MAIN -> HUMAN => stock -1
        if from_zone == "main" and current_zone == "human" and not state.get("taken", False):
            last_update_time, ok, msg = apply_stock_change(
                supa=supa,
                active_box_id=active_box_id,
                delta=-1,
                event_name=f"SAME GID TRANSFER MAIN->HUMAN (gid {gid})",
                last_update_time=last_update_time
            )
            if ok:
                sync_taken_state_related_gids(global_states, gid, True)
                take_delta += 1
                messages.append(msg)
                close_transfer_candidate(state)

        # HUMAN -> MAIN => stock +1
        elif from_zone == "human" and current_zone == "main" and state.get("taken", False):
            last_update_time, ok, msg = apply_stock_change(
                supa=supa,
                active_box_id=active_box_id,
                delta=+1,
                event_name=f"SAME GID TRANSFER HUMAN->MAIN (gid {gid})",
                last_update_time=last_update_time
            )
            if ok:
                sync_taken_state_related_gids(global_states, gid, False)
                return_delta += 1
                messages.append(msg)
                close_transfer_candidate(state)

    return last_update_time, take_delta, return_delta, messages


def resolve_transfer_candidates(
    global_states,
    frame_idx,
    transfer_window_frames,
    supa,
    active_box_id,
    last_update_time,
):
    messages = []
    take_delta = 0
    return_delta = 0

    candidate_gids = []
    target_gids = []

    for gid, state in global_states.items():
        if candidate_is_alive(state, frame_idx, transfer_window_frames):
            candidate_gids.append(gid)

        if not state.get("is_ghost", False):
            if state.get("last_stable_zone") in ("main", "human"):
                if state.get("zone_stable_frames", 0) >= MIN_STABLE_FRAMES_BEFORE_TRANSFER:
                    target_gids.append(gid)

    used_candidates = set()
    used_targets = set()

    for candidate_gid in candidate_gids:
        candidate_state = global_states[candidate_gid]
        from_zone = candidate_state["transfer_from_zone"]
        to_zone = expected_target_zone(from_zone)
        if to_zone is None:
            continue

        best_target_gid = None
        best_dist = 1e18

        anchor = candidate_state.get("transfer_anchor_point")
        if anchor is None:
            continue

        for target_gid in target_gids:
            if target_gid == candidate_gid:
                continue
            if target_gid in used_targets:
                continue

            target_state = global_states[target_gid]

            if target_state.get("last_stable_zone") != to_zone:
                continue

            if from_zone == "main" and (candidate_state.get("taken", False) or target_state.get("taken", False)):
                continue

            if from_zone == "human" and not candidate_state.get("taken", False):
                continue

            target_point = target_state.get("last_bottom_center")
            if target_point is None:
                continue

            dist = euclidean_dist(anchor, target_point)
            if dist <= (GHOST_MATCH_RADIUS * 2.0) and dist < best_dist:
                best_dist = dist
                best_target_gid = target_gid

        if best_target_gid is None:
            continue

        target_state = global_states[best_target_gid]

        if from_zone == "main" and target_state["last_stable_zone"] == "human" and not candidate_state["taken"] and not target_state["taken"]:
            last_update_time, ok, msg = apply_stock_change(
                supa=supa,
                active_box_id=active_box_id,
                delta=-1,
                event_name=f"CROSS GID TRANSFER MAIN->HUMAN (gid {candidate_gid}->{best_target_gid})",
                last_update_time=last_update_time
            )
            if ok:
                sync_taken_state_related_gids(global_states, candidate_gid, True)
                sync_taken_state_related_gids(global_states, best_target_gid, True)

                take_delta += 1
                messages.append(msg)
                used_candidates.add(candidate_gid)
                used_targets.add(best_target_gid)
                close_transfer_candidate(candidate_state)

        elif from_zone == "human" and target_state["last_stable_zone"] == "main" and candidate_state["taken"]:
            last_update_time, ok, msg = apply_stock_change(
                supa=supa,
                active_box_id=active_box_id,
                delta=+1,
                event_name=f"CROSS GID TRANSFER HUMAN->MAIN (gid {candidate_gid}->{best_target_gid})",
                last_update_time=last_update_time
            )
            if ok:
                sync_taken_state_related_gids(global_states, candidate_gid, False)
                sync_taken_state_related_gids(global_states, best_target_gid, False)

                return_delta += 1
                messages.append(msg)
                used_candidates.add(candidate_gid)
                used_targets.add(best_target_gid)
                close_transfer_candidate(candidate_state)

    return last_update_time, take_delta, return_delta, messages


# ===================== MAIN =====================
def main():
    model = YOLO(MODEL_PATH)
    supa = SupabaseREST(SUPABASE_URL, SUPABASE_KEY)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        write_state(last_event=f"Cannot open video: {VIDEO_PATH}")
        print("Cannot open video:", VIDEO_PATH)
        return

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6 or fps > 240:
        fps = 30.0

    ghost_max_gap_frames = safe_int_frames(GHOST_MAX_MISSING_SECONDS, fps)
    global_ttl_frames = safe_int_frames(GLOBAL_TTL_SECONDS, fps)
    track_ttl_frames = safe_int_frames(TRACK_TTL_SECONDS, fps)
    transfer_window_frames = safe_int_frames(TRANSFER_WINDOW_SECONDS, fps)

    print(f"[INFO] FPS: {fps:.2f}")
    print(f"[INFO] GHOST_MAX_GAP_FRAMES: {ghost_max_gap_frames}")
    print(f"[INFO] GLOBAL_TTL_FRAMES: {global_ttl_frames}")
    print(f"[INFO] TRACK_TTL_FRAMES: {track_ttl_frames}")
    print(f"[INFO] GHOST_MATCH_RADIUS: {GHOST_MATCH_RADIUS}")
    print(f"[INFO] TAKEN_SYNC_RADIUS: {TAKEN_SYNC_RADIUS}")
    print(f"[INFO] TRANSFER_WINDOW_FRAMES: {transfer_window_frames}")

    global_states = {}
    tracker_to_global = {}
    tracker_last_seen = {}
    next_global_id_ref = [1]

    last_update_time = 0.0

    active_box_id = None
    last_qr_text = None
    frame_idx = 0

    take_count = 0
    return_count = 0
    ghost_match_count = 0
    new_global_count = 0
    missing_count = 0
    transfer_candidate_count = 0
    transfer_resolved_count = 0
    same_gid_transfer_count = 0
    cross_gid_transfer_count = 0

    write_state(active_box_id=None, last_event="CV running...", last_decrement=None)

    while True:
        ret, frame = cap.read()
        if not ret:
            write_state(
                active_box_id=active_box_id,
                last_event="CV finished (end of video).",
                last_decrement=None
            )
            break

        frame_idx += 1

        # ===================== QR READ =====================
        if frame_idx % QR_READ_EVERY_N_FRAMES == 0:
            y1, y2, x1, x2 = QR_ROI

            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height, y2))
            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width, x2))

            roi = frame[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else frame

            qr_text = decode_qr_pyzbar(roi)

            if SHOW_QR_DEBUG:
                _, threshold = preprocess_for_qr(roi)
                cv2.imshow("QR ROI", roi)
                cv2.imshow("QR THRESH", threshold)

            if qr_text:
                try:
                    new_id = parse_box_id_from_qr(qr_text)
                    if new_id != active_box_id:
                        active_box_id = new_id
                        last_qr_text = qr_text
                        write_state(
                            active_box_id=active_box_id,
                            last_event=f"QR detected: {new_id}",
                            last_decrement=None
                        )
                except Exception as e:
                    write_state(
                        active_box_id=active_box_id,
                        last_event=f"QR Parse Error: {str(e)[:100]}",
                        last_decrement=None
                    )

        # ===================== TRACKING =====================
        results = model.track(
            frame,
            tracker="bytetrack.yaml",
            conf=CONF_THRESHOLD,
            persist=True,
            verbose=False
        )

        mx1, my1, mx2, my2 = MAIN_BOX
        hx1, hy1, hx2, hy2 = HUMAN_BOX

        cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 0, 255), 2)
        cv2.putText(frame, "MAIN_BOX", (mx1, max(30, my1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 255), 2)
        cv2.putText(frame, "HUMAN_BOX", (hx1, max(30, hy1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        cv2.putText(frame,
                    f"Active QR ID: {active_box_id if active_box_id else '-'}",
                    (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (200, 200, 255),
                    2)

        if last_qr_text:
            cv2.putText(frame,
                        f"QR: {last_qr_text[:45]}",
                        (40, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (200, 200, 255),
                        2)

        active_global_ids_this_frame = set()

        if results:
            result = results[0]
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xywh.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)

                for (x, y, w, h), conf, tid in zip(boxes, confs, track_ids):
                    px = int(x)
                    py = int(y + (h / 2))

                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    x2 = int(x + w / 2)
                    y2 = int(y + h / 2)

                    observed_zone = get_zone(px, py, MAIN_BOX, HUMAN_BOX, margin=ZONE_MARGIN)
                    det_bottom_center = bottom_center_from_xywh(x, y, w, h)

                    gid, assign_mode, matched_dist = assign_global_id_simple(
                        tid=tid,
                        det_bottom_center=det_bottom_center,
                        tracker_to_global=tracker_to_global,
                        global_states=global_states,
                        active_global_ids_this_frame=active_global_ids_this_frame,
                        next_global_id_ref=next_global_id_ref,
                        current_frame_idx=frame_idx,
                        ghost_max_gap_frames=ghost_max_gap_frames,
                        ghost_match_radius=GHOST_MATCH_RADIUS
                    )

                    if gid not in global_states:
                        init_zone = "outside" if observed_zone == "ambiguous" else observed_zone
                        global_states[gid] = init_global_state(frame_idx, initial_zone=init_zone)
                        new_global_count += 1

                    state = global_states[gid]

                    if assign_mode == "ghost_match":
                        ghost_match_count += 1

                    tracker_last_seen[tid] = frame_idx
                    state["last_seen_frame"] = frame_idx
                    state["last_bottom_center"] = det_bottom_center
                    state["last_bbox"] = xywh_to_xyxy(x, y, w, h)
                    state["is_ghost"] = False
                    state["ghost_since_frame"] = None

                    if state["current_tracker_id"] != tid:
                        if tid not in state["tracker_ids"]:
                            state["tracker_ids"].append(tid)
                        state["current_tracker_id"] = tid

                    stable_changed, prev_zone, stable_zone = update_zone_stability(
                        state,
                        observed_zone
                    )

                    if observed_zone == stable_zone and observed_zone != "ambiguous":
                        state["zone_stable_frames"] += 1
                    else:
                        if stable_changed:
                            state["zone_stable_frames"] = 1

                    if state.get("transfer_candidate", False) and stable_zone == state.get("transfer_from_zone"):
                        close_transfer_candidate(state)

                    color_assign = (0, 255, 0)
                    if assign_mode == "ghost_match":
                        color_assign = (0, 255, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_assign, 2)
                    cv2.circle(frame, (px, py), 4, (255, 0, 0), -1)

                    text1 = f"tid={tid} gid={gid} {assign_mode}"
                    if matched_dist is not None:
                        text1 += f" d={matched_dist:.1f}"

                    cv2.putText(
                        frame,
                        text1,
                        (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        color_assign,
                        2
                    )
                    cv2.putText(
                        frame,
                        f"obs={observed_zone} stable={stable_zone} stableN={state['zone_stable_frames']} taken={state['taken']}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        (255, 255, 0),
                        1
                    )

        # ===================== HANDLE MISSING -> OPEN TRANSFER CANDIDATE =====================
        for gid, state in global_states.items():
            if gid in active_global_ids_this_frame:
                continue

            if state["last_seen_frame"] is None:
                continue

            gap = frame_idx - state["last_seen_frame"]

            if gap >= 1 and not state["is_ghost"]:
                state["is_ghost"] = True
                if state["ghost_since_frame"] is None:
                    state["ghost_since_frame"] = frame_idx
                missing_count += 1

                opened = open_transfer_candidate(state, frame_idx)
                if opened:
                    transfer_candidate_count += 1

            if state.get("transfer_candidate", False):
                if not candidate_is_alive(state, frame_idx, transfer_window_frames):
                    close_transfer_candidate(state)

            if SHOW_GHOST_BOX and state["is_ghost"] and state["last_bbox"] is not None and gap <= ghost_max_gap_frames:
                gx1, gy1, gx2, gy2 = to_int_box(state["last_bbox"])
                ghost_color = (0, 165, 255)

                if state.get("transfer_candidate", False):
                    ghost_color = (255, 255, 0)

                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), ghost_color, 2)

                if state["last_bottom_center"] is not None:
                    gcx, gcy = map(int, state["last_bottom_center"])
                    cv2.circle(frame, (gcx, gcy), 4, ghost_color, -1)

                text = f"GHOST gid={gid} gap={gap}/{ghost_max_gap_frames}"
                if state.get("transfer_candidate", False):
                    text += f" TRANSFER_FROM={state['transfer_from_zone']}"

                cv2.putText(
                    frame,
                    text,
                    (gx1, max(20, gy1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    ghost_color,
                    2
                )

        # ===================== RESOLVE SAME-GID TRANSFER =====================
        last_update_time, take_delta_1, return_delta_1, msgs_1 = resolve_same_gid_transfer(
            global_states=global_states,
            frame_idx=frame_idx,
            transfer_window_frames=transfer_window_frames,
            supa=supa,
            active_box_id=active_box_id,
            last_update_time=last_update_time,
        )
        take_count += take_delta_1
        return_count += return_delta_1
        if take_delta_1 > 0 or return_delta_1 > 0:
            same_gid_transfer_count += (take_delta_1 + return_delta_1)
            transfer_resolved_count += (take_delta_1 + return_delta_1)
            for msg in msgs_1:
                print(msg)

        # ===================== RESOLVE CROSS-GID TRANSFER =====================
        last_update_time, take_delta_2, return_delta_2, msgs_2 = resolve_transfer_candidates(
            global_states=global_states,
            frame_idx=frame_idx,
            transfer_window_frames=transfer_window_frames,
            supa=supa,
            active_box_id=active_box_id,
            last_update_time=last_update_time,
        )
        take_count += take_delta_2
        return_count += return_delta_2
        if take_delta_2 > 0 or return_delta_2 > 0:
            cross_gid_transfer_count += (take_delta_2 + return_delta_2)
            transfer_resolved_count += (take_delta_2 + return_delta_2)
            for msg in msgs_2:
                print(msg)

        cleanup_old_tracker_mappings(
            tracker_to_global=tracker_to_global,
            tracker_last_seen=tracker_last_seen,
            current_frame_idx=frame_idx,
            ttl_frames=track_ttl_frames
        )

        cleanup_old_globals(
            global_states=global_states,
            current_frame_idx=frame_idx,
            ttl_frames=global_ttl_frames
        )

        # Overlay summary
        cv2.putText(frame, f"Take count (-1): {take_count}", (40, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)
        cv2.putText(frame, f"Return count (+1): {return_count}", (40, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)
        cv2.putText(frame, f"Ghost match: {ghost_match_count}", (40, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)
        cv2.putText(frame, f"Transfer candidates: {transfer_candidate_count}", (40, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)
        cv2.putText(frame, f"Transfer resolved: {transfer_resolved_count}", (40, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)
        cv2.putText(frame, f"Same GID transfer: {same_gid_transfer_count}", (40, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
        cv2.putText(frame, f"Cross GID transfer: {cross_gid_transfer_count}", (40, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
        cv2.putText(frame, f"Global alive: {len(global_states)}", (40, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
        cv2.putText(frame, f"Ghost radius: {GHOST_MATCH_RADIUS}px", (40, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
        cv2.putText(frame, f"Taken sync radius: {TAKEN_SYNC_RADIUS}px", (40, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
        cv2.putText(frame, f"Transfer window: {TRANSFER_WINDOW_SECONDS:.1f}s", (40, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)

        if SHOW_MAIN_WINDOW:
            cv2.imshow("CV Counting Candidate Transfer (Taken Sync Fixed)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            write_state(
                active_box_id=active_box_id,
                last_event="CV stopped by user.",
                last_decrement=None
            )
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()