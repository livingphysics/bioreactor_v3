"""
Relay Control GUI (uses bioreactor relay driver)

A simple GUI to control relays via the bioreactor component system.
Reads relay names and pins from config, uses RelayDriver for all IO.
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os

# Add parent directory to path to allow imports from src
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from src import Bioreactor, Config


class RelayControlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Relay Control")

        # Initialize bioreactor with only relays enabled
        override = {k: False for k in Config.INIT_COMPONENTS}
        override['relays'] = True
        Config.INIT_COMPONENTS = override

        try:
            self.bio = Bioreactor(Config)
        except Exception as e:
            messagebox.showerror("Init Error", f"Failed to initialize bioreactor:\n{e}")
            root.destroy()
            return

        if not self.bio.is_component_initialized('relays'):
            messagebox.showerror("Relay Error",
                "Relays failed to initialize.\n\nCheck:\n"
                "- lgpio is installed\n"
                "- Running with proper permissions\n"
                "- GPIO pins are available")
            root.destroy()
            return

        self.driver = self.bio.relay_driver
        self.relay_names = self.driver.relay_names
        self.state_labels = {}

        self.root.geometry(f"420x{60 + 50 * len(self.relay_names)}")
        self._create_widgets()
        self._update_states()

    def _create_widgets(self):
        title = tk.Label(self.root, text="Relay Control", font=("Helvetica", 14, "bold"))
        title.pack(pady=(10, 5))

        for name in self.relay_names:
            frame = tk.Frame(self.root)
            frame.pack(pady=4, padx=10, fill='x')

            label = tk.Label(frame, text=name, width=12, anchor='w',
                             font=("Helvetica", 11))
            label.pack(side='left')

            state_label = tk.Label(frame, text="OFF", width=6,
                                   bg='red', fg='white',
                                   font=("Helvetica", 10, "bold"))
            state_label.pack(side='left', padx=5)
            self.state_labels[name] = state_label

            btn_frame = tk.Frame(frame)
            btn_frame.pack(side='left', padx=5)

            tk.Button(btn_frame, text="ON", width=6,
                      command=lambda n=name: self._set(n, True)).pack(side='left', padx=2)
            tk.Button(btn_frame, text="OFF", width=6,
                      command=lambda n=name: self._set(n, False)).pack(side='left', padx=2)
            tk.Button(btn_frame, text="TOGGLE", width=8,
                      command=lambda n=name: self._toggle(n)).pack(side='left', padx=2)

        # All-off button at the bottom
        tk.Button(self.root, text="ALL OFF", width=20,
                  command=self._all_off).pack(pady=10)

    def _set(self, name, state):
        self.driver.set(name, state)
        self._update_states()

    def _toggle(self, name):
        self.driver.toggle(name)
        self._update_states()

    def _all_off(self):
        self.driver.off_all()
        self._update_states()

    def _update_states(self):
        states = self.driver.get_all_states()
        for name, label in self.state_labels.items():
            on = states.get(name, False)
            label.config(text="ON" if on else "OFF",
                         bg='green' if on else 'red')

    def on_closing(self):
        self.driver.off_all()
        try:
            self.bio.finish()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = RelayControlGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
