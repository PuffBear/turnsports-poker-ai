# 🎮 Web-Based Poker GUI

## ✨ Features
- **Beautiful dark-themed interface** with realistic poker table
- **Real-time gameplay** using WebSockets  
- **AI Coach integration** with instant advice
- **Smooth animations** and professional card design
- **Responsive layout** works on desktop and tablet

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install flask flask-socketio
```

### 2. Run the Server
```bash
cd gui/web
python app.py
```

### 3. Open Browser
Navigate to: **http://localhost:5000**

## 🎯 How to Play

1. **Click "New Hand"** to start
2. **See your cards** at the bottom
3. **Use action buttons** to play (Fold, Call, Raise, etc.)
4. **Click "Get Advice"** for AI coach recommendation
5. **Coach shows:**
   - Recommended action
   - Equity percentage
   - Pot odds
   - Strategic analysis

## 🎨 Design Highlights

- **Green felt poker table** with realistic shadows
- **Professional card styling** with suit colors
- **Modern glassmorphism** for coach panel
- **Smooth transitions** and hover effects
- **Dark theme** optimized for long sessions

## 🤖 Bot Training Status

**Note:** Training results show the RL bot is struggling. This is common in poker due to:
- Massive state space
- Sparse rewards
- Complex strategy

**For best experience:**
- Bot plays randomly (entertaining but beatable)
- **Coach is the star** - use it to learn strategy!
- Focus on the coaching advice, not bot strength

## 🎓 Learning Tool

This GUI is perfect for:
- **Learning poker strategy** from the coach
- **Understanding pot odds** and equity
- **Seeing game theory** in action
- **Demo/portfolio piece** showing agentic AI

## 🔧 Troubleshooting

**"Connection refused"**: Make sure Flask server is running

**"No module named flask"**: Run `pip install flask flask-socketio`

**Bot plays weirdly**: Expected - RL training needs more work. Coach is reliable!

---

**Enjoy the game! Focus on the coaching - that's where the magic happens!** 🎰✨
