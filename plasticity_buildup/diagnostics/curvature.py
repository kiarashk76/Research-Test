import torch


def ntk_matrix(model, x):
    parameters = tuple(model.parameters())
    outputs = model(x).reshape(x.shape[0], -1)
    sample_jacobians = []
    for sample_index in range(outputs.shape[0]):
        output_jacobians = []
        for output_index in range(outputs.shape[1]):
            gradients = torch.autograd.grad(outputs[sample_index, output_index], parameters, retain_graph=True, allow_unused=True)
            flat = torch.cat([
                gradient.reshape(-1) if gradient is not None else torch.zeros_like(parameter).reshape(-1)
                for parameter, gradient in zip(parameters, gradients)
            ])
            output_jacobians.append(flat)
        sample_jacobians.append(torch.stack(output_jacobians))
    jacobian = torch.stack(sample_jacobians)
    return torch.einsum("aop,bop->ab", jacobian, jacobian)


def ntk_change(model, x, reference_ntk, relative=True):
    current = ntk_matrix(model, x)
    change = torch.linalg.norm(current - reference_ntk)
    if relative:
        change = change / torch.linalg.norm(reference_ntk).clamp_min(1e-12)
    return {"global": change, "per_layer": None}


def hessian_spectrum(model, x, y, loss_fn):
    parameters = tuple(model.parameters())
    loss = loss_fn(model(x), y)
    first_gradients = torch.autograd.grad(loss, parameters, create_graph=True)
    flat_gradient = torch.cat([gradient.reshape(-1) for gradient in first_gradients])
    rows = []
    for index in range(flat_gradient.numel()):
        second = torch.autograd.grad(flat_gradient[index], parameters, retain_graph=True, allow_unused=True)
        rows.append(torch.cat([
            gradient.reshape(-1) if gradient is not None else torch.zeros_like(parameter).reshape(-1)
            for parameter, gradient in zip(parameters, second)
        ]))
    hessian = torch.stack(rows)
    eigenvalues = torch.linalg.eigvalsh(0.5 * (hessian + hessian.T))
    return {
        "global": eigenvalues[-1],
        "per_layer": None,
        "details": {
            "eigenvalues": eigenvalues,
            "largest": eigenvalues[-1],
            "smallest": eigenvalues[0],
            "trace": eigenvalues.sum(),
            "negative_fraction": (eigenvalues < 0).float().mean(),
        },
    }
