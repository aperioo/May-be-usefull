import tkinter as tk
import pyautogui
import threading


class MouseTracker:
    def __init__(self, root):
        self.root = root
        self.root.title("Mouse Tracker")

        self.label = tk.Label(root, text="Mouse position: (x, y)")
        self.label.pack(padx=20, pady=20)

        self.update_position()

    def update_position(self):
        x, y = pyautogui.position()
        self.label.config(text=f"Mouse position: ({x}, {y})")
        self.root.after(100, self.update_position)  # Update the position every 100 milliseconds


def main():
    root = tk.Tk()
    tracker = MouseTracker(root)
    root.mainloop()


if __name__ == "__main__":
    main()
