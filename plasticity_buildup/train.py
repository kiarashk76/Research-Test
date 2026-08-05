import numpy as np


def train_on_task(model, optimizer, method, x, y, loss_fn, num_epochs):
    """Train on one complete synthetic task without mini-batches."""
    method.before_task(x, y)
    losses = []
    model.train()
    for _ in range(num_epochs):
        method.before_update(x, y)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = loss_fn(prediction, y)
        loss.backward()
        optimizer.step()
        method.after_update(x, y)
        losses.append(loss.item())
    method.after_task(x, y)
    return np.asarray(losses, dtype=np.float64)
