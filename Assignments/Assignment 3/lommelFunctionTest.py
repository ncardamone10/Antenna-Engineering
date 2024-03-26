import mpmath
import numpy as np
import matplotlib.pyplot as plt

# Set the precision for mpmath calculations
mpmath.mp.dps = 25

# Define the range of x values
x = np.linspace(0, 100, 1000)

# Define specific values for v and u
v_values = [1,2,3]  # replace with your values
u_values = [1]  # replace with your values

# Calculate the corresponding y values using mpmath and plot the results
for v in v_values:
    for u in u_values:
        y = [mpmath.lommels1(u, v, xi, zeroprec=64, infprec=64) for xi in x]  # increase maxprec
        plt.plot(x, (y), label=f'u={u}, v={v}')

plt.xlabel('x')
plt.ylabel('lommels1(x)')
plt.title('Plot of mpmath.lommels1')
plt.legend()
plt.grid(True)
plt.show()