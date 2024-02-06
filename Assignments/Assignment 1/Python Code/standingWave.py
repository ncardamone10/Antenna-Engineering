import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
length = 60  # Length of the transmission line
points = 500  # Number of points along the line
frequency = 0.5  # Frequency of the wave
omega = 2 * np.pi * frequency  # Angular frequency
wavelength = 10
k = 2 * np.pi / wavelength

# Time settings for the animation
duration = 5  # Duration of the animation in seconds
frames = 100  # Number of frames in the animation

# Create the transmission line space
x = np.linspace(0, length, points)

# Initialize the figure and axis
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(0, length)
ax.set_ylim(-2, 2)  # Adjust the y-axis limits to fit the wave amplitude
ax.set_title("Standing Wave on a Transmission Line")
ax.set_xlabel("Position along the line")
ax.set_ylabel("Amplitude")
ax.grid(True)

# Initialize the frame
def init():
    line.set_data([], [])
    return line,

# Animation update function
def update(frame):
    t = frame / frames * duration
    y = np.cos(omega * t) * np.sin(k * x )
    line.set_data(x, y)
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=duration/frames*1000)

# Display the animation
plt.show()
