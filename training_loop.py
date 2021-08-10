# this won't run, it's a simplified version showing the main logic

def get_grads(agreement_threshold, 
              batch_size, 
              loss_fn,
              n_agreement_envs, 
              params, 
              output,
              target,
              scale_grad_inverse_sparsity,
              ):
    """
    Use the and mask to put gradients together.
    Modifies gradients wrt params in place (inside param.grad).
    Returns mean loss and masks for diagnostics.

    Args:
        agreement_threshold: a float between 0 and 1 (tau in the paper).
            If 1 -> requires complete sign agreement across all environments (everything else gets masked out),
             if 0 it requires no agreement, and it becomes essentially standard sgd if method == 'and_mask'. Values
             in between are fractional ratios of agreement.
        batch_size: The original batch size per environment. Needed to perform reshaping, so that grads can be computed
            independently per each environment.
        loss_fn: the loss function
        n_agreement_envs: the number of environments that were stacked in the inputs. Needed to perform reshaping.
        params: the model parameters
        output: the output of the model, where inputs were *all examples from all envs stacked in a big batch*. This is
            done to at least compute the forward pass somewhat efficiently.
        scale_grad_inverse_sparsity: If True, rescale the magnitude of the gradient components that survived the mask,
            layer-wise, to compensate for the reduce overall magnitude after masking and/or geometric mean.

    Returns:
        mean_loss: mean loss across environments
        masks: a list of the binary masks (every element corresponds to one layer) applied to the gradient.
    """

    param_gradients = [[] for _ in params]
    outputs = output.view(n_agreement_envs, batch_size, -1)
    targets = target.view(n_agreement_envs, batch_size, -1)

    outputs = outputs.squeeze(-1)
    targets = targets.squeeze(-1)

    # forward step
    total_loss = 0.
    for env_outputs, env_targets in zip(outputs, targets):
        env_loss = loss_fn(env_outputs, env_targets)
        total_loss += env_loss
        env_grads = torch.autograd.grad(env_loss, params,
                                           retain_graph=True)
        for grads, env_grad in zip(param_gradients, env_grads):
            grads.append(env_grad)
    mean_loss = total_loss / n_agreement_envs
    assert len(param_gradients) == len(params)
    assert len(param_gradients[0]) == n_agreement_envs

    # backward step: aggregate gradients using mask and update weights
    masks = []
    avg_grads = []
    weights = []
    for param, grads in zip(params, param_gradients):
        assert len(grads) == n_agreement_envs
        grads = torch.stack(grads, dim=0)
        assert grads.shape == (n_agreement_envs,) + param.shape
        grad_signs = torch.sign(grads)
        mask = torch.mean(grad_signs, dim=0).abs() >= agreement_threshold
        mask = mask.to(torch.float32)
        assert mask.numel() == param.numel()
        avg_grad = torch.mean(grads, dim=0)
        assert mask.shape == avg_grad.shape

        mask_t = (mask.sum() / mask.numel())
        param.grad = mask * avg_grad
        if scale_grad_inverse_sparsity:
            param.grad *= (1. / (1e-10 + mask_t))

        weights.append(param.data)
        avg_grads.append(avg_grad)
        masks.append(mask)

    return mean_loss, masks

def trainig_loop():
    train_loader, test_loader = get_train_test_loader()

    model = get_model()
    optimizer = Adam()
    criterion = 
    for x, y in train_loader:
        x, y = batch

        model.train()
        optimizer.zero_grad()

        y_pred = model(x)

        # The "batch_size" in this function refers to the batch size per env
        # Since we treat every example as one env, we should set the parameter
        # n_agreement_envs equal to batch size
        mean_loss, masks  = get_grads(
                agreement_threshold=config['agreement_threshold'], # e.g. 
                batch_size=1,
                loss_fn=CrossEntropyLoss()
                n_agreement_envs=80,
                params=optimizer.param_groups[0]['params'],
                output=y_pred,
                target=y,
                scale_grad_inverse_sparsity=True,
        )

        optimizer.step()
