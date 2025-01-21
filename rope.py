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
    """
     freqs = torch.pow(theta, -torch.arange(0, head_dim, 2, device=device)[:(head_dim//2)].float() / head_dim)
    pos = torch.arange(seqlen, device=device).float()[:max_seq_len]

    freqs = torch.outer(freqs, pos).transpose(-2, -1).float()  # (head_dim // 2, max_seq_len)
    freqs = reshape_for_broadcast(freqs, query_real)

    # shape: (batch_size, seqlen, n_local_heads, head_dim // 2)
    query_rotated_real = freqs.cos() * query_real - freqs.sin() * query_imag
    query_rotated_imag = freqs.sin() * query_real + freqs.cos() * query_imag
    key_rotated_real = freqs.cos() * key_real - freqs.sin() * key_imag
    key_rotated_imag = freqs.sin() * key_real + freqs.cos() * key_imag

    # Then, combine these trigonometric values with the tensors query_real, query_imag,
    # key_real, and key_imag.
    query_stack = torch.stack((query_rotated_real, query_rotated_imag), dim=-1)
    key_stack = torch.stack((key_rotated_real, key_rotated_imag), dim=-1)

    query_out = query_stack.reshape(query.shape)
    key_out = key_stack.reshape(key.shape)

    
    Apply rotary embeddings to input tensors using the given frequency tensor.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_heads, head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_heads, head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
        theta (float): Base scaling factor for sinusoidal frequency.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Modified query and key tensors with rotary embeddings.
    """
    
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

'''
def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_heads, head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_heads, head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
        theta (float): Base scaling factor for sinusoidal frequency.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Modified query and key tensors with rotary embeddings.
    """
    _, seqlen, _, _ = query.shape
    device = query.device
    
    # Generate frequencies
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device) / head_dim))
    positions = torch.arange(0, seqlen, device=device)  
    freqs = torch.einsum('i,j->ij', positions, inv_freq)  # Outer product: (max_seq_len, head_dim // 2)
    
    # Create sine and cosine embeddings
    freqs_cis = torch.stack((freqs.sin(), freqs.cos()), dim=-1)  # Shape: (max_seq_len, head_dim // 2, 2)
    freqs_cis = freqs_cis.reshape(seqlen, -1)  # Flatten last dimension for complex representation
    
    # Truncate frequencies to sequence length
    
    
    # Reshape frequency tensor for broadcasting
    freqs_cis = reshape_for_broadcast(freqs_cis, query)
    
    # Reshape query and key tensors to separate real and imaginary components
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    
    # Apply rotary embeddings
    print('q_real',query_real)
    print('q_imag',query_imag)
    query_out_real = query_real * freqs_cis[..., 1] - query_imag * freqs_cis[..., 0]
    query_out_imag = query_real * freqs_cis[..., 0] + query_imag * freqs_cis[..., 1]
    key_out_real = key_real * freqs_cis[..., 1] - key_imag * freqs_cis[..., 0]
    key_out_imag = key_real * freqs_cis[..., 0] + key_imag * freqs_cis[..., 1]
    print('query_out_real',query_out_real)
    print('query_out_imag',query_out_imag)
    print('key_out_real',key_out_real)
    print('key_out_imag',key_out_imag)
    # Combine real and imaginary parts back into the final tensors
    query_out = torch.stack((query_out_real, query_out_imag), dim=-1).flatten(-2)
    key_out = torch.stack((key_out_real, key_out_imag), dim=-1).flatten(-2)
    
    return query_out, key_out
'''
