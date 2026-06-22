<<<<<<< HEAD
import json
import time
import threading
import tkinter as tk
from tkinter import font, messagebox
import requests

# ===================== CONFIG =====================
SUPABASE_URL = "https://yrsdedmvnpavswfjfmdy.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlyc2RlZG12bnBhdnN3ZmpmbWR5Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2OTE2MzI1NSwiZXhwIjoyMDg0NzM5MjU1fQ.d_EatbYxL22Tr4eTqXagf8LdF7IqWX82EUOtXfnOsYo"
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
        root.geometry("920x600") # 稍微加高一點空間給控制面板
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

        self.alerted_boxes = set() 
        
        self.manual_active_id = None 
        
        self._build_control_panel()

        self._tick()

    def _build_control_panel(self):
        control_frame = tk.Frame(self.root, bg="#1e293b", highlightthickness=1, highlightbackground="#334155")
        control_frame.pack(fill="x", side="bottom", padx=18, pady=18)
        
        tk.Label(control_frame, text="🔧 手動控制面板:", fg="white", bg="#1e293b", font=self.f_sub).pack(side="left", padx=10, pady=10)

        # 動態產生鎖定按鈕
        for box_id in BOX_IDS:
            btn = tk.Button(control_frame, text=f"鎖定 {box_id}", bg="#3b82f6", fg="white", bd=0, padx=10, pady=5,
                            command=lambda b=box_id: self._set_manual_id(b))
            btn.pack(side="left", padx=5)

        # === 新增：增加庫存按鈕 (綠色) ===
        add_btn = tk.Button(control_frame, text="增加 (+1)", bg="#10b981", fg="white", bd=0, padx=15, pady=5, font=self.f_sub,
                               command=self._manual_add)
        add_btn.pack(side="right", padx=10)

        # 扣除庫存按鈕 (紅色)
        deduct_btn = tk.Button(control_frame, text="扣除 (-1)", bg="#ef4444", fg="white", bd=0, padx=15, pady=5, font=self.f_sub,
                               command=self._manual_deduct)
        deduct_btn.pack(side="right", padx=5)
        
        # 解除鎖定按鈕 (灰色)
        unlock_btn = tk.Button(control_frame, text="解除鎖定", bg="#64748b", fg="white", bd=0, padx=10, pady=5,
                               command=lambda: self._set_manual_id(None))
        unlock_btn.pack(side="right", padx=10)

    def _set_manual_id(self, box_id):
        self.manual_active_id = box_id
        if box_id:
            print(f"已手動鎖定 ID: {box_id}")
        else:
            print("已解除手動鎖定，恢復自動模式")

    def _manual_deduct(self):
        if not self.manual_active_id:
            messagebox.showwarning("操作錯誤", "請先點擊左側按鈕「鎖定」一個箱子！")
            return
            
        stock_str = self.card_widgets[self.manual_active_id]["lbl_stock"].cget("text")
        try:
            current_stock = int(stock_str)
        except ValueError:
            messagebox.showerror("錯誤", "目前尚未讀取到庫存資料，請稍候。")
            return
            
        if current_stock <= 0:
            messagebox.showwarning("警告", "庫存已經是 0，無法再扣除了！")
            return
            
        new_stock = current_stock - 1
        
        def worker():
            try:
                update_url = f"{self.supa.base_url}/rest/v1/{TABLE}?{ID_FIELD}=eq.{self.manual_active_id}"
                payload = {STOCK_FIELD: new_stock}
                headers = self.supa.headers()
                headers["Prefer"] = "return=representation" 
                
                print(f"發送手動扣除請求... 目標: {self.manual_active_id}, 新庫存: {new_stock}")
                res = requests.patch(update_url, headers=headers, json=payload)
                if res.ok:
                    print("✅ 手動扣除成功！")
                else:
                    print(f"❌ 扣除失敗: {res.text}")
            except Exception as e:
                print(f"❌ 發生錯誤: {e}")
                
        threading.Thread(target=worker, daemon=True).start()

    # === 新增：手動增加庫存功能 ===
    def _manual_add(self):
        if not self.manual_active_id:
            messagebox.showwarning("操作錯誤", "請先點擊左側按鈕「鎖定」一個箱子！")
            return
            
        stock_str = self.card_widgets[self.manual_active_id]["lbl_stock"].cget("text")
        try:
            current_stock = int(stock_str)
        except ValueError:
            messagebox.showerror("錯誤", "目前尚未讀取到庫存資料，請稍候。")
            return
            
        new_stock = current_stock + 1  # 這裡改成 +1
        
        def worker():
            try:
                update_url = f"{self.supa.base_url}/rest/v1/{TABLE}?{ID_FIELD}=eq.{self.manual_active_id}"
                payload = {STOCK_FIELD: new_stock}
                headers = self.supa.headers()
                headers["Prefer"] = "return=representation" 
                
                print(f"發送手動增加請求... 目標: {self.manual_active_id}, 新庫存: {new_stock}")
                res = requests.patch(update_url, headers=headers, json=payload)
                if res.ok:
                    print("✅ 手動增加成功！")
                else:
                    print(f"❌ 增加失敗: {res.text}")
            except Exception as e:
                print(f"❌ 發生錯誤: {e}")
                
        threading.Thread(target=worker, daemon=True).start()

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
            
            stock_value = r.get(STOCK_FIELD, "-")
            w["lbl_stock"].configure(text=str(stock_value))
            w["lbl_cat"].configure(text=f"Category: {r.get(CATEGORY_FIELD, '-')}")

            #================ 警告與自動補貨邏輯 ================
            try:
                stock_num = int(stock_value)
                
                # 判斷這個箱子是不是目前的 active_id，且庫存 <= 0
                if box_id == active_id and stock_num <= 0: 
                    
                    if box_id not in self.alerted_boxes:
                        self.alerted_boxes.add(box_id)
                        messagebox.showwarning("庫存警告", f"注意：ID {box_id} 的庫存只剩下 {stock_num}！按下確定補貨 +1。")
                        
                        new_stock = stock_num + 1
                        try:
                            update_url = f"{self.supa.base_url}/rest/v1/{TABLE}?{ID_FIELD}=eq.{box_id}"
                            payload = {STOCK_FIELD: new_stock}
                            
                            headers = self.supa.headers()
                            headers["Prefer"] = "return=representation" 
                            
                            print(f"\n--- 準備發送更新 ---")
                            print(f"目標網址: {update_url}") 
                            print(f"更新內容: {payload}")
                            
                            response = requests.patch(update_url, headers=headers, json=payload)
                            
                            if not response.ok:
                                print(f"❌ 更新失敗！狀態碼: {response.status_code}")
                                print(f"錯誤訊息: {response.text}")
                            else:
                                print(f"✅ 成功！已「單獨」將 ID {box_id} 庫存更新為 {new_stock}")
                                
                        except Exception as e:
                            print(f"❌ 發生其他異常: {e}")
                            
                else:
                    if box_id in self.alerted_boxes:
                        self.alerted_boxes.remove(box_id)
                        
            except ValueError:
                pass
            #================================================

            if active_id is not None and box_id == active_id:
                w["card"].configure(highlightbackground="#2dd4bf")
            else:
                w["card"].configure(highlightbackground="#121826")

    def _tick(self):
        # 讀取相機狀態
        cam_active_id, status = read_state()
        
        # 如果有手動鎖定，就覆蓋掉相機的狀態
        if self.manual_active_id is not None:
            active_id = self.manual_active_id
            status = "Manual Override "
        else:
            active_id = cam_active_id

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
import time
import threading
import tkinter as tk
from tkinter import font, messagebox
import requests

# ===================== CONFIG =====================
SUPABASE_URL = "https://yrsdedmvnpavswfjfmdy.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlyc2RlZG12bnBhdnN3ZmpmbWR5Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2OTE2MzI1NSwiZXhwIjoyMDg0NzM5MjU1fQ.d_EatbYxL22Tr4eTqXagf8LdF7IqWX82EUOtXfnOsYo"
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
        root.geometry("920x600") # 稍微加高一點空間給控制面板
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

        self.alerted_boxes = set() 
        
        self.manual_active_id = None 
        
        self._build_control_panel()

        self._tick()

    def _build_control_panel(self):
        control_frame = tk.Frame(self.root, bg="#1e293b", highlightthickness=1, highlightbackground="#334155")
        control_frame.pack(fill="x", side="bottom", padx=18, pady=18)
        
        tk.Label(control_frame, text="🔧 手動控制面板:", fg="white", bg="#1e293b", font=self.f_sub).pack(side="left", padx=10, pady=10)

        # 動態產生鎖定按鈕
        for box_id in BOX_IDS:
            btn = tk.Button(control_frame, text=f"鎖定 {box_id}", bg="#3b82f6", fg="white", bd=0, padx=10, pady=5,
                            command=lambda b=box_id: self._set_manual_id(b))
            btn.pack(side="left", padx=5)

        # === 新增：增加庫存按鈕 (綠色) ===
        add_btn = tk.Button(control_frame, text="增加 (+1)", bg="#10b981", fg="white", bd=0, padx=15, pady=5, font=self.f_sub,
                               command=self._manual_add)
        add_btn.pack(side="right", padx=10)

        # 扣除庫存按鈕 (紅色)
        deduct_btn = tk.Button(control_frame, text="扣除 (-1)", bg="#ef4444", fg="white", bd=0, padx=15, pady=5, font=self.f_sub,
                               command=self._manual_deduct)
        deduct_btn.pack(side="right", padx=5)
        
        # 解除鎖定按鈕 (灰色)
        unlock_btn = tk.Button(control_frame, text="解除鎖定", bg="#64748b", fg="white", bd=0, padx=10, pady=5,
                               command=lambda: self._set_manual_id(None))
        unlock_btn.pack(side="right", padx=10)

    def _set_manual_id(self, box_id):
        self.manual_active_id = box_id
        if box_id:
            print(f"已手動鎖定 ID: {box_id}")
        else:
            print("已解除手動鎖定，恢復自動模式")

    def _manual_deduct(self):
        if not self.manual_active_id:
            messagebox.showwarning("操作錯誤", "請先點擊左側按鈕「鎖定」一個箱子！")
            return
            
        stock_str = self.card_widgets[self.manual_active_id]["lbl_stock"].cget("text")
        try:
            current_stock = int(stock_str)
        except ValueError:
            messagebox.showerror("錯誤", "目前尚未讀取到庫存資料，請稍候。")
            return
            
        if current_stock <= 0:
            messagebox.showwarning("警告", "庫存已經是 0，無法再扣除了！")
            return
            
        new_stock = current_stock - 1
        
        def worker():
            try:
                update_url = f"{self.supa.base_url}/rest/v1/{TABLE}?{ID_FIELD}=eq.{self.manual_active_id}"
                payload = {STOCK_FIELD: new_stock}
                headers = self.supa.headers()
                headers["Prefer"] = "return=representation" 
                
                print(f"發送手動扣除請求... 目標: {self.manual_active_id}, 新庫存: {new_stock}")
                res = requests.patch(update_url, headers=headers, json=payload)
                if res.ok:
                    print("✅ 手動扣除成功！")
                else:
                    print(f"❌ 扣除失敗: {res.text}")
            except Exception as e:
                print(f"❌ 發生錯誤: {e}")
                
        threading.Thread(target=worker, daemon=True).start()

    # === 新增：手動增加庫存功能 ===
    def _manual_add(self):
        if not self.manual_active_id:
            messagebox.showwarning("操作錯誤", "請先點擊左側按鈕「鎖定」一個箱子！")
            return
            
        stock_str = self.card_widgets[self.manual_active_id]["lbl_stock"].cget("text")
        try:
            current_stock = int(stock_str)
        except ValueError:
            messagebox.showerror("錯誤", "目前尚未讀取到庫存資料，請稍候。")
            return
            
        new_stock = current_stock + 1  # 這裡改成 +1
        
        def worker():
            try:
                update_url = f"{self.supa.base_url}/rest/v1/{TABLE}?{ID_FIELD}=eq.{self.manual_active_id}"
                payload = {STOCK_FIELD: new_stock}
                headers = self.supa.headers()
                headers["Prefer"] = "return=representation" 
                
                print(f"發送手動增加請求... 目標: {self.manual_active_id}, 新庫存: {new_stock}")
                res = requests.patch(update_url, headers=headers, json=payload)
                if res.ok:
                    print("✅ 手動增加成功！")
                else:
                    print(f"❌ 增加失敗: {res.text}")
            except Exception as e:
                print(f"❌ 發生錯誤: {e}")
                
        threading.Thread(target=worker, daemon=True).start()

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
            
            stock_value = r.get(STOCK_FIELD, "-")
            w["lbl_stock"].configure(text=str(stock_value))
            w["lbl_cat"].configure(text=f"Category: {r.get(CATEGORY_FIELD, '-')}")

            #================ 警告與自動補貨邏輯 ================
            try:
                stock_num = int(stock_value)
                
                # 判斷這個箱子是不是目前的 active_id，且庫存 <= 0
                if box_id == active_id and stock_num <= 0: 
                    
                    if box_id not in self.alerted_boxes:
                        self.alerted_boxes.add(box_id)
                        messagebox.showwarning("庫存警告", f"注意：ID {box_id} 的庫存只剩下 {stock_num}！按下確定補貨 +1。")
                        
                        new_stock = stock_num + 1
                        try:
                            update_url = f"{self.supa.base_url}/rest/v1/{TABLE}?{ID_FIELD}=eq.{box_id}"
                            payload = {STOCK_FIELD: new_stock}
                            
                            headers = self.supa.headers()
                            headers["Prefer"] = "return=representation" 
                            
                            print(f"\n--- 準備發送更新 ---")
                            print(f"目標網址: {update_url}") 
                            print(f"更新內容: {payload}")
                            
                            response = requests.patch(update_url, headers=headers, json=payload)
                            
                            if not response.ok:
                                print(f"❌ 更新失敗！狀態碼: {response.status_code}")
                                print(f"錯誤訊息: {response.text}")
                            else:
                                print(f"✅ 成功！已「單獨」將 ID {box_id} 庫存更新為 {new_stock}")
                                
                        except Exception as e:
                            print(f"❌ 發生其他異常: {e}")
                            
                else:
                    if box_id in self.alerted_boxes:
                        self.alerted_boxes.remove(box_id)
                        
            except ValueError:
                pass
            #================================================

            if active_id is not None and box_id == active_id:
                w["card"].configure(highlightbackground="#2dd4bf")
            else:
                w["card"].configure(highlightbackground="#121826")

    def _tick(self):
        # 讀取相機狀態
        cam_active_id, status = read_state()
        
        # 如果有手動鎖定，就覆蓋掉相機的狀態
        if self.manual_active_id is not None:
            active_id = self.manual_active_id
            status = "Manual Override "
        else:
            active_id = cam_active_id

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