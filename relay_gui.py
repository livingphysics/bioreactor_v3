import tkinter as tk
from tkinter import messagebox
from actuate_relays import actuate_relay, get_relay_states, cleanup_gpio, is_gpio_initialized

class RelayGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Relay Control")
        self.root.geometry("400x300")
        
        # Verify GPIO is initialized
        if not is_gpio_initialized():
            messagebox.showerror("GPIO Error", 
                "GPIO chip not initialized!\n\nCheck:\n- lgpio is installed\n- Running with proper permissions\n- GPIO pins are available")
            root.destroy()
            return
        
        self.relays = ['relay1', 'relay2', 'relay3', 'relay4']
        self.buttons = {}
        self.state_labels = {}
        
        self.create_widgets()
        self.update_states()
    
    def create_widgets(self):
        for i, relay in enumerate(self.relays):
            frame = tk.Frame(self.root)
            frame.pack(pady=5, padx=10, fill='x')
            
            # Relay label
            label = tk.Label(frame, text=relay.upper(), width=10, anchor='w')
            label.pack(side='left')
            
            # State label
            state_label = tk.Label(frame, text="OFF", width=8, bg='red')
            state_label.pack(side='left', padx=5)
            self.state_labels[relay] = state_label
            
            # Buttons
            btn_frame = tk.Frame(frame)
            btn_frame.pack(side='left')
            
            tk.Button(btn_frame, text="ON", width=6, 
                     command=lambda r=relay: self.set_relay(r, True)).pack(side='left', padx=2)
            tk.Button(btn_frame, text="OFF", width=6,
                     command=lambda r=relay: self.set_relay(r, False)).pack(side='left', padx=2)
            tk.Button(btn_frame, text="TOGGLE", width=8,
                     command=lambda r=relay: self.toggle_relay(r)).pack(side='left', padx=2)
    
    def set_relay(self, relay, state):
        actuate_relay(relay, state)
        self.update_states()
    
    def toggle_relay(self, relay):
        states = get_relay_states()
        current = states.get(relay, False)
        self.set_relay(relay, not current)
    
    def update_states(self):
        states = get_relay_states()
        for relay, label in self.state_labels.items():
            state = states.get(relay, False)
            label.config(text="ON" if state else "OFF", 
                        bg='green' if state else 'red')
    
    def on_closing(self):
        cleanup_gpio()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = RelayGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

