import numpy as np
import matplotlib.pyplot as plt

x = np.array([-1, 1, 1.5, 3])
y = np.array([0.5, 2, 4, 3])

m = 1
c = 1
alpha = 0.05

plot_iterations = 500

loss_values = []

def loss_function(x, y, m, c):
    n = len(x)
    y_pred = m * x + c
    loss = (1/n) * np.sum((y - y_pred) ** 2)
    return loss

def plot_line(m, c, ax, label, linestyle='-'):
    x_vals = np.linspace(-2, 3, 100)
    y_vals = m * x_vals + c
    ax.plot(x_vals, y_vals, linestyle=linestyle, label=label)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax2.scatter(x, y, color='red', label='Data points')

plot_line(m, c, ax2, label='m0 = 1, c0 = 1')

def gradient_function(x, y, m, c, alpha, iterations):
    n = len(x)
    for i in range(iterations):
        y_pred = m * x + c
        dm = (-2/n) * np.sum(x * (y - y_pred))
        dc = (-2/n) * np.sum(y - y_pred)
        m -= alpha * dm
        c -= alpha * dc
        loss = loss_function(x, y, m, c)
        loss_values.append(loss)
        print(f"m{i+1} = {round(m,4)}, c{i+1} = {round(c,4)} , Loss = {round(loss,4)}")
        if i < plot_iterations:
            plot_line(m, c, ax2, label=f'')
    return m, c

m_final, c_final = gradient_function(x, y, m, c, alpha, plot_iterations)
plot_line(m_final, c_final, ax2, label='Best fit line', linestyle='--')

ax1.plot(range(1, plot_iterations + 1), loss_values, linestyle='-', color='blue')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Loss')
ax1.set_title('Loss function over iterations')

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Line fitting over iterations')
ax2.legend()

plt.tight_layout()
plt.show()
