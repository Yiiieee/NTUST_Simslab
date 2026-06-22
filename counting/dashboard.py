import json
import time
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

BOX_IDS = [260101, 260104, 260123, 260124] # item 物料  
UI_REFRESH_SECONDS = 1 

STATE_FILE = "state.json" #這個的目的是 讓dashboard和counter_service能夠共享當前的active_box_id和last_event狀態
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
        root.geometry("920x520")
        root.configure(bg="#0b0f14")

        self.f_title = font.Font(family="Arial", size=18, weight="bold")
        self.f_sub = font.Font(family="Arial", size=11)
        self.f_stock = font.Font(family="Arial", size=28, weight="bold")
        self.f_id = font.Font(family="Arial", size=12, weight="bold")
        self.f_cat = font.Font(family="Arial", size=10)

        header = tk.Frame(root, bg="#0b0f14")
        header.pack(fill="x", padx=18, pady=(16, 8))

        tk.Label(header, text="Inventory Stock Dashboard", fg="white", bg="#0b0f14", font=self.f_title).pack(anchor="w")

        self.detected_var = tk.StringVar(value="Active QR ID: -")
        self.status_var = tk.StringVar(value="Status: starting...")
        tk.Label(header, textvariable=self.detected_var, fg="#8aa4ff", bg="#0b0f14", font=self.f_sub).pack(anchor="w", pady=(6, 0))
        tk.Label(header, textvariable=self.status_var, fg="#9aa7b2", bg="#0b0f14", font=self.f_sub).pack(anchor="w", pady=(2, 0))

        self.cards_frame = tk.Frame(root, bg="#0b0f14")
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

        self._tick()

    def _make_card(self, parent, box_id: int):
        card = tk.Frame(parent, bg="#121826", bd=0, highlightthickness=2, highlightbackground="#121826")
        top = tk.Frame(card, bg="#121826")
        top.pack(fill="x", padx=14, pady=(12, 6))

        tk.Label(top, text=f"ID {box_id}", fg="white", bg="#121826", font=self.f_id).pack(anchor="w")

        lbl_cat = tk.Label(card, text="Category: -", fg="#9aa7b2", bg="#121826", font=self.f_cat)
        lbl_cat.pack(anchor="w", padx=14)

        lbl_stock = tk.Label(card, text="-", fg="white", bg="#121826", font=self.f_stock)
        lbl_stock.pack(anchor="w", padx=14, pady=(10, 2))

        tk.Label(card, text="stock", fg="#9aa7b2", bg="#121826", font=self.f_cat).pack(anchor="w", padx=14, pady=(0, 12))

        self.card_widgets[box_id] = {"card": card, "lbl_cat": lbl_cat, "lbl_stock": lbl_stock}
        return card

    def _render_rows(self, rows, active_id, status):
        self.detected_var.set(f"Active QR ID: {active_id if active_id else '-'}")
        self.status_var.set(f"Status: {status}")

        by_id = {int(r.get(ID_FIELD)): r for r in rows}
        for box_id, w in self.card_widgets.items():
            r = by_id.get(box_id, {})
            w["lbl_stock"].configure(text=str(r.get(STOCK_FIELD, "-")))
            w["lbl_cat"].configure(text=f"Category: {r.get(CATEGORY_FIELD, '-')}")

            #================
            
            
            #================
            if active_id is not None and box_id == active_id:
                w["card"].configure(highlightbackground="#2dd4bf")
            else:
                w["card"].configure(highlightbackground="#121826")

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
    main()
