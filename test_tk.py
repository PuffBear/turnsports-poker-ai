#!/usr/bin/env python3
"""Minimal Tk test to verify Tkinter is working."""
import tkinter as tk

root = tk.Tk()
root.title("Tk Test")
root.geometry("400x300")
root.configure(bg='white')

# Bright test labels
tk.Label(root, text="TEST 1: Red background", bg='red', fg='white', font=("Arial", 16)).pack(pady=10)
tk.Label(root, text="TEST 2: Blue background", bg='blue', fg='white', font=("Arial", 16)).pack(pady=10)
tk.Label(root, text="TEST 3: Green background", bg='green', fg='white', font=("Arial", 16)).pack(pady=10)
tk.Label(root, text="If you see this, Tkinter works!", bg='yellow', fg='black', font=("Arial", 20, "bold")).pack(pady=20)

print("Tk test window created. Close it to exit.")
print(f"Tk version: {root.tk.call('info', 'patchlevel')}")

root.mainloop()

