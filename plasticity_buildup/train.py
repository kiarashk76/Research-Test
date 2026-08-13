import numpy as np


def train_on_task(model, optimizer, method, x, y, loss_fn, num_epochs):
    """Train on one complete synthetic task without mini-batches."""
    method.prepare_for_task(x, y)
    method.before_task(x, y)
    losses = []
    model.train()
    for _ in range(num_epochs):
        method.before_update(x, y)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        task_loss = loss_fn(prediction, y)
        loss = method.modify_loss(task_loss, x, y)
        loss.backward()
        method.after_backward(x, y)
        optimizer.step()
        method.after_update(x, y)
        losses.append(loss.item())
    method.after_task(x, y)
    return np.asarray(losses, dtype=np.float64)
