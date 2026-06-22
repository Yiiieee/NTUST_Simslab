<<<<<<< HEAD
import json
import threading
import tkinter as tk
from tkinter import font
import requests

# ===================== CONFIG =====================
SUPABASE_URL = "https://yrsdedmvnpavswfjfmdy.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlyc2RlZG12bnBhdnN3ZmpmbWR5Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2OTE2MzI1NSwiZXhwIjoyMDg0NzM5MjU1fQ.d_EatbYxL22Tr4eTqXagf8LdF7IqWX82EUOtXfnOsYo"  # anon key + RLS policy aman (JANGAN service_role)
TABLE = "Boxes"
ID_FIELD = "id"
STOCK_FIELD = "Stock"
CATEGORY_FIELD = "Category"

BOX_IDS = [260101, 260102, 260103, 260104, 260123, 260124]
UI_REFRESH_SECONDS = 1

STATE_FILE = "state.json"
# ==================================================


class SupabaseREST:
    def __init__(self, base_url: str, key: str):
        self.base_url = base_url.rstrip("/")
        self.key = (key or "").strip()

    def headers(self):
        h = {"Accept": "application/json"}
        if self.key:
            h["apikey"] = self.key
            h["Authorization"] = f"Bearer {self.key}"
        return h

    def fetch_boxes(self, ids):
        ids_csv = ",".join(str(i) for i in ids)
        url = f"{self.base_url}/rest/v1/{TABLE}?{ID_FIELD}=in.({ids_csv})&select=*"
        r = requests.get(url, headers=self.headers(), timeout=8)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            raise ValueError("Response bukan list.")
        data.sort(key=lambda x: x.get(ID_FIELD, 0))
        return data


def read_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            s = json.load(f)
        return s.get("active_box_id"), s.get("last_event", "Ready")
    except Exception:
        return None, "Ready"


class Dashboard:
    def __init__(self, root: tk.Tk, supa: SupabaseREST):
        self.root = root
        self.supa = supa

        root.title("Stock Dashboard")
        root.geometry("920x560")
        root.configure(bg="#0b0f14")

        self.f_title = font.Font(family="Arial", size=18, weight="bold")
        self.f_sub = font.Font(family="Arial", size=11)
        self.f_stock = font.Font(family="Arial", size=28, weight="bold")
        self.f_id = font.Font(family="Arial", size=12, weight="bold")
        self.f_cat = font.Font(family="Arial", size=10)
        self.f_alert = font.Font(family="Arial", size=11, weight="bold")
        self.f_card_alert = font.Font(family="Arial", size=10, weight="bold")

        # Base colors
        self.bg_app = "#0b0f14"
        self.card_normal = "#121826"
        self.border_normal = "#121826"
        self.border_active = "#2dd4bf"

        # Animated alert colors
        self.card_alert_colors = ["#7f1d1d", "#b91c1c"]   # dark red <-> red
        self.banner_alert_colors = ["#991b1b", "#dc2626"] # dark red <-> bright red
        self.label_alert_colors = ["#b91c1c", "#ef4444"]  # red shades

        self.alert_blink_state = False
        self.anim_job = None
        self.zero_stock_boxes = set()

        header = tk.Frame(root, bg=self.bg_app)
        header.pack(fill="x", padx=18, pady=(16, 8))

        tk.Label(
            header,
            text="Inventory Stock Dashboard",
            fg="white",
            bg=self.bg_app,
            font=self.f_title
        ).pack(anchor="w")

        self.detected_var = tk.StringVar(value="Active QR ID: -")
        self.status_var = tk.StringVar(value="Status: starting...")
        self.alert_var = tk.StringVar(value="")

        tk.Label(
            header,
            textvariable=self.detected_var,
            fg="#8aa4ff",
            bg=self.bg_app,
            font=self.f_sub
        ).pack(anchor="w", pady=(6, 0))

        tk.Label(
            header,
            textvariable=self.status_var,
            fg="#9aa7b2",
            bg=self.bg_app,
            font=self.f_sub
        ).pack(anchor="w", pady=(2, 0))

        self.alert_banner = tk.Label(
            header,
            textvariable=self.alert_var,
            fg="white",
            bg=self.banner_alert_colors[1],
            font=self.f_alert,
            padx=10,
            pady=10,
            anchor="w"
        )
        self.alert_banner.pack(anchor="w", pady=(8, 0), fill="x")
        self.alert_banner.pack_forget()

        self.cards_frame = tk.Frame(root, bg=self.bg_app)
        self.cards_frame.pack(fill="both", expand=True, padx=18, pady=10)

        self.card_widgets = {}
        cols = 3
        for idx, box_id in enumerate(BOX_IDS):
            r = idx // cols
            c = idx % cols
            card = self._make_card(self.cards_frame, box_id)
            card.grid(row=r, column=c, padx=10, pady=10, sticky="nsew")

        for c in range(cols):
            self.cards_frame.grid_columnconfigure(c, weight=1)
        for r in range((len(BOX_IDS) + cols - 1) // cols):
            self.cards_frame.grid_rowconfigure(r, weight=1)

        self._animate_alerts()
        self._tick()

    def _make_card(self, parent, box_id: int):
        card = tk.Frame(
            parent,
            bg=self.card_normal,
            bd=0,
            highlightthickness=3,
            highlightbackground=self.border_normal
        )

        top = tk.Frame(card, bg=self.card_normal)
        top.pack(fill="x", padx=14, pady=(12, 6))

        lbl_id = tk.Label(
            top,
            text=f"ID {box_id}",
            fg="white",
            bg=self.card_normal,
            font=self.f_id
        )
        lbl_id.pack(anchor="w")

        lbl_cat = tk.Label(
            card,
            text="Category: -",
            fg="#cbd5e1",
            bg=self.card_normal,
            font=self.f_cat
        )
        lbl_cat.pack(anchor="w", padx=14)

        lbl_stock = tk.Label(
            card,
            text="-",
            fg="white",
            bg=self.card_normal,
            font=self.f_stock
        )
        lbl_stock.pack(anchor="w", padx=14, pady=(10, 2))

        lbl_stock_text = tk.Label(
            card,
            text="stock",
            fg="#cbd5e1",
            bg=self.card_normal,
            font=self.f_cat
        )
        lbl_stock_text.pack(anchor="w", padx=14, pady=(0, 8))

        lbl_alert = tk.Label(
            card,
            text="⚠ OUT OF STOCK",
            fg="white",
            bg=self.label_alert_colors[1],
            font=self.f_card_alert,
            padx=8,
            pady=6
        )
        lbl_alert.pack(anchor="w", padx=14, pady=(0, 12), fill="x")
        lbl_alert.pack_forget()

        self.card_widgets[box_id] = {
            "card": card,
            "top": top,
            "lbl_id": lbl_id,
            "lbl_cat": lbl_cat,
            "lbl_stock": lbl_stock,
            "lbl_stock_text": lbl_stock_text,
            "lbl_alert": lbl_alert,
        }
        return card

    def _apply_card_colors(self, widgets, bg_color, border_color, is_zero_stock):
        widgets["card"].configure(bg=bg_color, highlightbackground=border_color)
        widgets["top"].configure(bg=bg_color)
        widgets["lbl_id"].configure(bg=bg_color)
        widgets["lbl_cat"].configure(bg=bg_color)
        widgets["lbl_stock"].configure(bg=bg_color)
        widgets["lbl_stock_text"].configure(bg=bg_color)

        if is_zero_stock:
            widgets["lbl_cat"].configure(fg="#fee2e2")
            widgets["lbl_stock"].configure(fg="white")
            widgets["lbl_stock_text"].configure(fg="#fee2e2")
        else:
            widgets["lbl_cat"].configure(fg="#cbd5e1")
            widgets["lbl_stock"].configure(fg="white")
            widgets["lbl_stock_text"].configure(fg="#cbd5e1")

    def _set_card_theme(self, box_id, widgets, is_zero_stock: bool, is_active: bool):
        if is_zero_stock:
            bg_color = self.card_alert_colors[1 if self.alert_blink_state else 0]
        else:
            bg_color = self.card_normal

        border_color = self.border_active if is_active else self.border_normal
        self._apply_card_colors(widgets, bg_color, border_color, is_zero_stock)

        if is_zero_stock:
            widgets["lbl_alert"].configure(
                bg=self.label_alert_colors[1 if self.alert_blink_state else 0],
                fg="white"
            )
            if not widgets["lbl_alert"].winfo_ismapped():
                widgets["lbl_alert"].pack(anchor="w", padx=14, pady=(0, 12), fill="x")
        else:
            if widgets["lbl_alert"].winfo_ismapped():
                widgets["lbl_alert"].pack_forget()

    def _animate_alerts(self):
        self.alert_blink_state = not self.alert_blink_state

        # animate banner
        if self.zero_stock_boxes:
            self.alert_banner.configure(
                bg=self.banner_alert_colors[1 if self.alert_blink_state else 0]
            )

        # animate each zero-stock card
        for box_id, widgets in self.card_widgets.items():
            is_zero_stock = box_id in self.zero_stock_boxes
            border_color = widgets["card"].cget("highlightbackground")
            if is_zero_stock:
                bg_color = self.card_alert_colors[1 if self.alert_blink_state else 0]
                self._apply_card_colors(widgets, bg_color, border_color, True)
                widgets["lbl_alert"].configure(
                    bg=self.label_alert_colors[1 if self.alert_blink_state else 0]
                )

        self.anim_job = self.root.after(500, self._animate_alerts)

    def _render_rows(self, rows, active_id, status):
        self.detected_var.set(f"Active QR ID: {active_id if active_id else '-'}")
        self.status_var.set(f"Status: {status}")

        by_id = {int(r.get(ID_FIELD)): r for r in rows}
        empty_boxes = []

        for box_id, w in self.card_widgets.items():
            r = by_id.get(box_id, {})
            stock_value = r.get(STOCK_FIELD, "-")
            category_value = r.get(CATEGORY_FIELD, "-")

            w["lbl_stock"].configure(text=str(stock_value))
            w["lbl_cat"].configure(text=f"Category: {category_value}")

            try:
                stock_num = int(stock_value)
            except (TypeError, ValueError):
                stock_num = None

            is_zero_stock = (stock_num == 0)
            is_active = (active_id is not None and box_id == active_id)

            if is_zero_stock:
                empty_boxes.append(str(box_id))

            self._set_card_theme(box_id, w, is_zero_stock, is_active)

        self.zero_stock_boxes = {int(x) for x in empty_boxes}

        if empty_boxes:
            self.alert_var.set(f"⚠ Alert: Out of stock in boxes {', '.join(empty_boxes)}")
            if not self.alert_banner.winfo_ismapped():
                self.alert_banner.pack(anchor="w", pady=(8, 0), fill="x")
        else:
            self.alert_var.set("")
            if self.alert_banner.winfo_ismapped():
                self.alert_banner.pack_forget()

    def _tick(self):
        active_id, status = read_state()

        def worker():
            try:
                rows = self.supa.fetch_boxes(BOX_IDS)
                self.root.after(0, lambda: self._render_rows(rows, active_id, status))
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Status: refresh error: {e}"))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(int(UI_REFRESH_SECONDS * 1000), self._tick)


def main():
    root = tk.Tk()
    supa = SupabaseREST(SUPABASE_URL, SUPABASE_KEY)
    Dashboard(root, supa)
    root.mainloop()


if __name__ == "__main__":
=======
import json
import threading
import tkinter as tk
from tkinter import font
import requests

# ===================== CONFIG =====================
SUPABASE_URL = "https://yrsdedmvnpavswfjfmdy.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlyc2RlZG12bnBhdnN3ZmpmbWR5Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2OTE2MzI1NSwiZXhwIjoyMDg0NzM5MjU1fQ.d_EatbYxL22Tr4eTqXagf8LdF7IqWX82EUOtXfnOsYo"  # anon key + RLS policy aman (JANGAN service_role)
TABLE = "Boxes"
ID_FIELD = "id"
STOCK_FIELD = "Stock"
CATEGORY_FIELD = "Category"

BOX_IDS = [260101, 260102, 260103, 260104, 260123, 260124]
UI_REFRESH_SECONDS = 1

STATE_FILE = "state.json"
# ==================================================


class SupabaseREST:
    def __init__(self, base_url: str, key: str):
        self.base_url = base_url.rstrip("/")
        self.key = (key or "").strip()

    def headers(self):
        h = {"Accept": "application/json"}
        if self.key:
            h["apikey"] = self.key
            h["Authorization"] = f"Bearer {self.key}"
        return h

    def fetch_boxes(self, ids):
        ids_csv = ",".join(str(i) for i in ids)
        url = f"{self.base_url}/rest/v1/{TABLE}?{ID_FIELD}=in.({ids_csv})&select=*"
        r = requests.get(url, headers=self.headers(), timeout=8)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            raise ValueError("Response bukan list.")
        data.sort(key=lambda x: x.get(ID_FIELD, 0))
        return data


def read_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            s = json.load(f)
        return s.get("active_box_id"), s.get("last_event", "Ready")
    except Exception:
        return None, "Ready"


class Dashboard:
    def __init__(self, root: tk.Tk, supa: SupabaseREST):
        self.root = root
        self.supa = supa

        root.title("Stock Dashboard")
        root.geometry("920x560")
        root.configure(bg="#0b0f14")

        self.f_title = font.Font(family="Arial", size=18, weight="bold")
        self.f_sub = font.Font(family="Arial", size=11)
        self.f_stock = font.Font(family="Arial", size=28, weight="bold")
        self.f_id = font.Font(family="Arial", size=12, weight="bold")
        self.f_cat = font.Font(family="Arial", size=10)
        self.f_alert = font.Font(family="Arial", size=11, weight="bold")
        self.f_card_alert = font.Font(family="Arial", size=10, weight="bold")

        # Base colors
        self.bg_app = "#0b0f14"
        self.card_normal = "#121826"
        self.border_normal = "#121826"
        self.border_active = "#2dd4bf"

        # Animated alert colors
        self.card_alert_colors = ["#7f1d1d", "#b91c1c"]   # dark red <-> red
        self.banner_alert_colors = ["#991b1b", "#dc2626"] # dark red <-> bright red
        self.label_alert_colors = ["#b91c1c", "#ef4444"]  # red shades

        self.alert_blink_state = False
        self.anim_job = None
        self.zero_stock_boxes = set()

        header = tk.Frame(root, bg=self.bg_app)
        header.pack(fill="x", padx=18, pady=(16, 8))

        tk.Label(
            header,
            text="Inventory Stock Dashboard",
            fg="white",
            bg=self.bg_app,
            font=self.f_title
        ).pack(anchor="w")

        self.detected_var = tk.StringVar(value="Active QR ID: -")
        self.status_var = tk.StringVar(value="Status: starting...")
        self.alert_var = tk.StringVar(value="")

        tk.Label(
            header,
            textvariable=self.detected_var,
            fg="#8aa4ff",
            bg=self.bg_app,
            font=self.f_sub
        ).pack(anchor="w", pady=(6, 0))

        tk.Label(
            header,
            textvariable=self.status_var,
            fg="#9aa7b2",
            bg=self.bg_app,
            font=self.f_sub
        ).pack(anchor="w", pady=(2, 0))

        self.alert_banner = tk.Label(
            header,
            textvariable=self.alert_var,
            fg="white",
            bg=self.banner_alert_colors[1],
            font=self.f_alert,
            padx=10,
            pady=10,
            anchor="w"
        )
        self.alert_banner.pack(anchor="w", pady=(8, 0), fill="x")
        self.alert_banner.pack_forget()

        self.cards_frame = tk.Frame(root, bg=self.bg_app)
        self.cards_frame.pack(fill="both", expand=True, padx=18, pady=10)

        self.card_widgets = {}
        cols = 3
        for idx, box_id in enumerate(BOX_IDS):
            r = idx // cols
            c = idx % cols
            card = self._make_card(self.cards_frame, box_id)
            card.grid(row=r, column=c, padx=10, pady=10, sticky="nsew")

        for c in range(cols):
            self.cards_frame.grid_columnconfigure(c, weight=1)
        for r in range((len(BOX_IDS) + cols - 1) // cols):
            self.cards_frame.grid_rowconfigure(r, weight=1)

        self._animate_alerts()
        self._tick()

    def _make_card(self, parent, box_id: int):
        card = tk.Frame(
            parent,
            bg=self.card_normal,
            bd=0,
            highlightthickness=3,
            highlightbackground=self.border_normal
        )

        top = tk.Frame(card, bg=self.card_normal)
        top.pack(fill="x", padx=14, pady=(12, 6))

        lbl_id = tk.Label(
            top,
            text=f"ID {box_id}",
            fg="white",
            bg=self.card_normal,
            font=self.f_id
        )
        lbl_id.pack(anchor="w")

        lbl_cat = tk.Label(
            card,
            text="Category: -",
            fg="#cbd5e1",
            bg=self.card_normal,
            font=self.f_cat
        )
        lbl_cat.pack(anchor="w", padx=14)

        lbl_stock = tk.Label(
            card,
            text="-",
            fg="white",
            bg=self.card_normal,
            font=self.f_stock
        )
        lbl_stock.pack(anchor="w", padx=14, pady=(10, 2))

        lbl_stock_text = tk.Label(
            card,
            text="stock",
            fg="#cbd5e1",
            bg=self.card_normal,
            font=self.f_cat
        )
        lbl_stock_text.pack(anchor="w", padx=14, pady=(0, 8))

        lbl_alert = tk.Label(
            card,
            text="⚠ OUT OF STOCK",
            fg="white",
            bg=self.label_alert_colors[1],
            font=self.f_card_alert,
            padx=8,
            pady=6
        )
        lbl_alert.pack(anchor="w", padx=14, pady=(0, 12), fill="x")
        lbl_alert.pack_forget()

        self.card_widgets[box_id] = {
            "card": card,
            "top": top,
            "lbl_id": lbl_id,
            "lbl_cat": lbl_cat,
            "lbl_stock": lbl_stock,
            "lbl_stock_text": lbl_stock_text,
            "lbl_alert": lbl_alert,
        }
        return card

    def _apply_card_colors(self, widgets, bg_color, border_color, is_zero_stock):
        widgets["card"].configure(bg=bg_color, highlightbackground=border_color)
        widgets["top"].configure(bg=bg_color)
        widgets["lbl_id"].configure(bg=bg_color)
        widgets["lbl_cat"].configure(bg=bg_color)
        widgets["lbl_stock"].configure(bg=bg_color)
        widgets["lbl_stock_text"].configure(bg=bg_color)

        if is_zero_stock:
            widgets["lbl_cat"].configure(fg="#fee2e2")
            widgets["lbl_stock"].configure(fg="white")
            widgets["lbl_stock_text"].configure(fg="#fee2e2")
        else:
            widgets["lbl_cat"].configure(fg="#cbd5e1")
            widgets["lbl_stock"].configure(fg="white")
            widgets["lbl_stock_text"].configure(fg="#cbd5e1")

    def _set_card_theme(self, box_id, widgets, is_zero_stock: bool, is_active: bool):
        if is_zero_stock:
            bg_color = self.card_alert_colors[1 if self.alert_blink_state else 0]
        else:
            bg_color = self.card_normal

        border_color = self.border_active if is_active else self.border_normal
        self._apply_card_colors(widgets, bg_color, border_color, is_zero_stock)

        if is_zero_stock:
            widgets["lbl_alert"].configure(
                bg=self.label_alert_colors[1 if self.alert_blink_state else 0],
                fg="white"
            )
            if not widgets["lbl_alert"].winfo_ismapped():
                widgets["lbl_alert"].pack(anchor="w", padx=14, pady=(0, 12), fill="x")
        else:
            if widgets["lbl_alert"].winfo_ismapped():
                widgets["lbl_alert"].pack_forget()

    def _animate_alerts(self):
        self.alert_blink_state = not self.alert_blink_state

        # animate banner
        if self.zero_stock_boxes:
            self.alert_banner.configure(
                bg=self.banner_alert_colors[1 if self.alert_blink_state else 0]
            )

        # animate each zero-stock card
        for box_id, widgets in self.card_widgets.items():
            is_zero_stock = box_id in self.zero_stock_boxes
            border_color = widgets["card"].cget("highlightbackground")
            if is_zero_stock:
                bg_color = self.card_alert_colors[1 if self.alert_blink_state else 0]
                self._apply_card_colors(widgets, bg_color, border_color, True)
                widgets["lbl_alert"].configure(
                    bg=self.label_alert_colors[1 if self.alert_blink_state else 0]
                )

        self.anim_job = self.root.after(500, self._animate_alerts)

    def _render_rows(self, rows, active_id, status):
        self.detected_var.set(f"Active QR ID: {active_id if active_id else '-'}")
        self.status_var.set(f"Status: {status}")

        by_id = {int(r.get(ID_FIELD)): r for r in rows}
        empty_boxes = []

        for box_id, w in self.card_widgets.items():
            r = by_id.get(box_id, {})
            stock_value = r.get(STOCK_FIELD, "-")
            category_value = r.get(CATEGORY_FIELD, "-")

            w["lbl_stock"].configure(text=str(stock_value))
            w["lbl_cat"].configure(text=f"Category: {category_value}")

            try:
                stock_num = int(stock_value)
            except (TypeError, ValueError):
                stock_num = None

            is_zero_stock = (stock_num == 0)
            is_active = (active_id is not None and box_id == active_id)

            if is_zero_stock:
                empty_boxes.append(str(box_id))

            self._set_card_theme(box_id, w, is_zero_stock, is_active)

        self.zero_stock_boxes = {int(x) for x in empty_boxes}

        if empty_boxes:
            self.alert_var.set(f"⚠ Alert: Out of stock in boxes {', '.join(empty_boxes)}")
            if not self.alert_banner.winfo_ismapped():
                self.alert_banner.pack(anchor="w", pady=(8, 0), fill="x")
        else:
            self.alert_var.set("")
            if self.alert_banner.winfo_ismapped():
                self.alert_banner.pack_forget()

    def _tick(self):
        active_id, status = read_state()

        def worker():
            try:
                rows = self.supa.fetch_boxes(BOX_IDS)
                self.root.after(0, lambda: self._render_rows(rows, active_id, status))
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Status: refresh error: {e}"))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(int(UI_REFRESH_SECONDS * 1000), self._tick)


def main():
    root = tk.Tk()
    supa = SupabaseREST(SUPABASE_URL, SUPABASE_KEY)
    Dashboard(root, supa)
    root.mainloop()


if __name__ == "__main__":
>>>>>>> 2a6bd7f1f941ef825654fcd4c38b276dd00248b6
    main()