from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0[")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0[")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, consider SparseAdam instead")

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['first_moment'] = torch.zeros_like(p.data)
                    state['second_moment'] = torch.zeros_like(p.data)

                step = state['step'] + 1
                beta1, beta2 = group['betas']
                lr, eps = group['lr'], group['eps']
                weight_decay = group['weight_decay']
                correct_bias = group['correct_bias']

                # Get first and second moments
                first_moment = state['first_moment']
                second_moment = state['second_moment']

                # Update first and second moments
                first_moment = beta1 * first_moment + (1 - beta1) * grad
                second_moment = beta2 * second_moment + (1 - beta2) * (grad ** 2)

                # Bias correction
                if correct_bias:
                    corrected_first_moment = first_moment / (1 - beta1**step)
                    corrected_second_moment = second_moment / (1 - beta2**step)
                else:
                    corrected_first_moment = first_moment
                    corrected_second_moment = second_moment
                
                # Apply weight decay
                p.data = p.data - lr * weight_decay * p.data
                p.data = p.data - lr * corrected_first_moment / (corrected_second_moment.sqrt() + eps)
               
                # Save updated moments and step
                state['step'] = step
                state['first_moment'] = first_moment
                state['second_moment'] = second_moment

        return loss
