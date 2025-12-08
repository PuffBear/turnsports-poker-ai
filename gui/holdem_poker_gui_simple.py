#!/usr/bin/env python3
"""Minimal poker GUI - only needs tkinter."""
import tkinter as tk

def main():
    root = tk.Tk()
    root.title("Texas Hold'em - 1v1 Table (Static Mock)")
    root.geometry("1000x650")
    root.configure(bg='#f5f5f5')
    
    # Title
    title = tk.Label(root, text="TEXAS HOLD'EM POKER TABLE", 
                     font=("Arial", 24, "bold"), bg='#f5f5f5', fg='black')
    title.pack(pady=10)
    
    # BOT section
    bot_frame = tk.Frame(root, bg='#e0e0e0', relief=tk.RAISED, borderwidth=2)
    bot_frame.pack(fill=tk.X, padx=40, pady=10)
    
    tk.Label(bot_frame, text="BOT", font=("Arial", 20, "bold"), 
             bg='#e0e0e0', fg='black').pack(pady=6)
    tk.Label(bot_frame, text="?? ??", font=("Courier", 28, "bold"), 
             bg='#e0e0e0', fg='black').pack()
    tk.Label(bot_frame, text="Stack: 100.0 BB", font=("Arial", 14), 
             bg='#e0e0e0', fg='black').pack(pady=4)
    
    # Board section
    board_frame = tk.Frame(root, bg='#d9f2d9', relief=tk.RAISED, borderwidth=2)
    board_frame.pack(fill=tk.X, padx=40, pady=10)
    
    tk.Label(board_frame, text="BOARD", font=("Arial", 18, "bold"), 
             bg='#d9f2d9', fg='black').pack(pady=6)
    tk.Label(board_frame, text="Ts 7c 2d Jc 5h", font=("Courier", 24), 
             bg='#d9f2d9', fg='black').pack()
    tk.Label(board_frame, text="Pot: 12.5 BB   |   Street: RIVER   |   To Call: 3.5 BB", 
             font=("Arial", 16, "bold"), bg='#d9f2d9', fg='blue').pack(pady=8)
    
    # Player section
    player_frame = tk.Frame(root, bg='#e0e0e0', relief=tk.RAISED, borderwidth=2)
    player_frame.pack(fill=tk.X, padx=40, pady=10)
    
    tk.Label(player_frame, text="YOU", font=("Arial", 20, "bold"), 
             bg='#e0e0e0', fg='black').pack(pady=6)
    tk.Label(player_frame, text="Ah Kd", font=("Courier", 28, "bold"), 
             bg='#e0e0e0', fg='black').pack()
    tk.Label(player_frame, text="Stack: 88.0 BB", font=("Arial", 14), 
             bg='#e0e0e0', fg='black').pack(pady=4)
    
    # Buttons
    btn_frame = tk.Frame(root, bg='#f5f5f5')
    btn_frame.pack(pady=20)
    
    tk.Button(btn_frame, text="Fold", width=12, height=2, 
              font=("Arial", 12)).pack(side=tk.LEFT, padx=10)
    tk.Button(btn_frame, text="Call", width=12, height=2, 
              font=("Arial", 12)).pack(side=tk.LEFT, padx=10)
    tk.Button(btn_frame, text="Raise", width=12, height=2, 
              font=("Arial", 12)).pack(side=tk.LEFT, padx=10)
    tk.Button(btn_frame, text="New Hand", width=12, height=2, 
              font=("Arial", 12)).pack(side=tk.LEFT, padx=10)
    
    # Status
    status = tk.Label(root, text="Table ready - Static mock (no gameplay in this file)", 
                      font=("Arial", 14), bg='#f5f5f5', fg='black')
    status.pack(pady=10)
    
    print("GUI created successfully!")
    print("If you see this window with all elements, Tkinter is working!")
    
    root.mainloop()

if __name__ == "__main__":
    main()

