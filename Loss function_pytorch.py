import torch
import matplotlib.pyplot as plt

x = torch.tensor([-1, 1, 1.5, 3], dtype=torch.float32).reshape(-1, 1)
y = torch.tensor([0.5, 2, 4, 3], dtype=torch.float32).reshape(-1, 1)

m = torch.tensor([1.0], requires_grad=True)
c = torch.tensor([1.0], requires_grad=True)
alpha = 0.05
iterations = 500
loss_values = []

def loss_function(y_pred, y):
    return ((y_pred - y) ** 2).mean()

def plot_line(m, c, ax, linestyle='-', label=None):
    x_vals = torch.linspace(-2, 3, 100).unsqueeze(1)
    y_vals = m * x_vals + c
    ax.plot(x_vals.detach().numpy(), y_vals.detach().numpy(), linestyle=linestyle, label=label)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax2.scatter(x.numpy(), y.numpy(), color='red', label='Data points')

for i in range(iterations):
    y_pred = m * x + c
    loss = loss_function(y_pred, y)
    loss_values.append(loss.item())
    loss.backward()

    with torch.no_grad():
        m -= alpha * m.grad
        c -= alpha * c.grad
        m.grad.zero_()
        c.grad.zero_()

    if i < 500:
        plot_line(m, c, ax2)

plot_line(m, c, ax2, linestyle='--', label='Best fit line')

ax1.plot(loss_values, color='blue')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Loss')
ax1.set_title('Loss over iterations')

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Line fitting over iterations')
ax2.legend()

plt.tight_layout()
plt.show()
