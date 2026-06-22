import torch
import subprocess
import sys
import os
import signal

def main():
    python_exec = sys.executable  # current Python interpreter

    # ensure paths are based on this script's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    counter_path = os.path.join(base_dir, "counting11.py")
    dashboard_path = os.path.join(base_dir, "dashboardv2.py")  # dashboard version

    # start both processes 
    p1 = subprocess.Popen([python_exec, counter_path])
    p2 = subprocess.Popen([python_exec, dashboard_path])

    print("RUNNING:")
    print(f" - counter_service.py (PID: {p1.pid})")
    print(f" - dashboard.py       (PID: {p2.pid})") 
    print("Press Ctrl+C to stop both processes.")

    try:
        # wait until one process stops
        while True:
            if p1.poll() is not None:
                print("counter_service stopped. Shutting down dashboard...")
                break
            if p2.poll() is not None:
                print("dashboard stopped. Shutting down counter_service...")
                break
    except KeyboardInterrupt:
        print("\nStopping...")

    # terminate both processes
    for p in (p1, p2):
        if p.poll() is None:
            try:
                if os.name == "nt":
                    p.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    p.terminate()
            except Exception:
                pass

    # wait briefly, then force kill if still alive
    for p in (p1, p2):
        try:
            p.wait(timeout=3)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

    print("Done.")

if __name__ == "__main__":
    main()