import customtkinter as ctk
import pandas as pd
import numpy as np
from tkinter import messagebox

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.cm as cm

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# Features & target
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Exit function
def exit_app():
    if messagebox.askokcancel("Exit", "Exit app?"):
        app.destroy()

# UI Setup
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

app = ctk.CTk()
app.geometry("1200x750")
app.title("Agro AI Dashboard")

# Sidebar
sidebar = ctk.CTkFrame(app, width=220)
sidebar.pack(side="left", fill="y")

ctk.CTkLabel(sidebar, text="Agro AI", font=("Arial", 20, "bold")).pack(pady=20)

active_btn = None

def animate_button(btn):
    def on_enter(e):
        btn.configure(fg_color="#2fa572")

    def on_leave(e):
        if btn != active_btn:
            btn.configure(fg_color=None)

    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)

def set_active(btn):
    global active_btn
    if active_btn:
        active_btn.configure(fg_color=None)
    btn.configure(fg_color="#1f6f4a")
    active_btn = btn

# Content frame
content = ctk.CTkFrame(app)
content.pack(side="right", expand=True, fill="both")

# Chart setup
fig, ax = plt.subplots(figsize=(5, 5))
canvas = FigureCanvasTkAgg(fig, master=content)
canvas.get_tk_widget().pack()

def draw_chart(labels, sizes, colors):
    ax.clear()
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    explode = [0.12 if i == 0 else 0 for i in range(len(labels))]

    wedges, _ = ax.pie(
        sizes,
        startangle=90,
        colors=colors,
        explode=explode,
        wedgeprops=dict(width=0.35, edgecolor="#1a1a1a")
    )

    wedges[0].set_facecolor("#00ff99")

    ax.text(0, 0, f"{labels[0]}\n{sizes[0]*100:.1f}%",
            ha="center", va="center", fontsize=16, color="#00ff99")

    legend_labels = [f"{l} - {s*100:.1f}%" for l, s in zip(labels, sizes)]

    legend = ax.legend(
        wedges, legend_labels,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        frameon=False
    )

    for text in legend.get_texts():
        text.set_color("white")

    ax.axis("off")
    canvas.draw()

# Input sliders
sliders = []
input_card = ctk.CTkFrame(content)
input_card.pack(side="left", padx=20, pady=20)

result = ctk.CTkLabel(content, text="Recommended Crop: ")
result.pack()

current_sizes = None
current_colors = None

def update_all():
    global current_sizes, current_colors

    vals = [s.get() for s in sliders]
    probs = model.predict_proba([vals])[0]

    data = list(zip(model.classes_, probs))
    data.sort(key=lambda x: x[1], reverse=True)

    labels = [d[0] for d in data]
    new_sizes = np.array([d[1] for d in data])

    result.configure(text=f"Recommended Crop: {labels[0]}")

    base_colors = cm.plasma(np.linspace(0, 1, len(labels)))

    if current_sizes is None:
        current_sizes = new_sizes
        current_colors = base_colors

    steps = 10
    for i in range(steps):
        interp_sizes = current_sizes + (new_sizes - current_sizes) * (i / steps)
        interp_colors = current_colors + (base_colors - current_colors) * (i / steps)

        draw_chart(labels, interp_sizes, interp_colors)
        app.update_idletasks()

    current_sizes = new_sizes
    current_colors = base_colors

def create_slider(label, f, t):
    frame = ctk.CTkFrame(input_card)
    frame.pack(pady=5, padx=10, fill="x")

    lbl = ctk.CTkLabel(frame, text=f"{label}: 0")
    lbl.pack(anchor="w")

    def update(v):
        lbl.configure(text=f"{label}: {int(v)}")
        update_all()

    s = ctk.CTkSlider(frame, from_=f, to=t, command=update)
    s.pack(fill="x")

    sliders.append(s)

# Create sliders
create_slider("Nitrogen", 0, 140)
create_slider("Phosphorus", 0, 145)
create_slider("Potassium", 0, 205)
create_slider("Temperature", 0, 50)
create_slider("Humidity", 0, 100)
create_slider("pH", 0, 14)
create_slider("Rainfall", 0, 300)

# Exit button
exit_btn = ctk.CTkButton(sidebar, text="Exit", fg_color="red", command=exit_app)
exit_btn.pack(pady=20, padx=10, fill="x")

animate_button(exit_btn)

# Run app
app.mainloop()