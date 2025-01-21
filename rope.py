from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)
def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    _, seqlen, _, _ = query.shape
    device = query.device
     # Reshape query and key tensors to separate real and imaginary components
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)

    # Generate frequencies
    inv_freq  = torch.pow(theta, -torch.arange(0, head_dim, 2, device=device)[:(head_dim//2)].float() / head_dim)
    positions = torch.arange(0, seqlen, device=device).float()[:max_seq_len]  # Use max_seq_len for full range
    freqs = torch.outer(inv_freq,positions).transpose(-2,-1).float()  # Outer product: (max_seq_len, head_dim // 2)
   
    freqs = reshape_for_broadcast(freqs, query_real)

   
   
    # Apply rotary embeddings
    query_out_real = freqs.cos()*query_real- freqs.sin()*query_imag
    query_out_imag = freqs.sin()*query_real+  freqs.cos()*query_imag
    key_out_real =  freqs.cos()*key_real - freqs.sin()*key_imag
    key_out_imag = freqs.sin()*key_real + freqs.cos()*key_imag 

    # Combine real and imaginary parts back into the final tensors
    query_st = torch.stack((query_out_real, query_out_imag), dim=-1)
    key_st = torch.stack((key_out_real, key_out_imag), dim=-1)
    query_out = query_st.reshape(query.shape)
    key_out = key_st.reshape(key.shape)
    return query_out, key_out


