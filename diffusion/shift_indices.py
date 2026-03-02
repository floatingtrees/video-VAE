import jax
import jax.numpy as jnp


def shift_indices_to_left(V, I_padded, dynamic_len):
    """Gather elements from V using indices I_padded, left-aligned with right padding.

    Args:
        V: Array of shape (n, ...) to gather from.
        I_padded: Integer array of shape (n,). First `dynamic_len` elements are valid
                  indices into V's first axis. Remaining elements are ignored (masked out).
        dynamic_len: Scalar integer, number of valid indices in I_padded.

    Returns:
        T: Array of shape (n, ...) where T[i] = V[I_padded[i]] for i < dynamic_len,
           and 0 for i >= dynamic_len.
        mask: Boolean array of shape (n,), True for valid positions (i < dynamic_len).
    """
    n = I_padded.shape[0]
    gathered = V[I_padded]
    mask = jnp.arange(n) < dynamic_len
    broadcast_mask = mask.reshape((-1,) + (1,) * (gathered.ndim - 1))
    T = jnp.where(broadcast_mask, gathered, 0)
    return T, mask


def convert_to_indices(binary_mask):
    """Convert a binary mask of 0s and 1s into a left-packed index array and dynamic length.

    E.g. [0, 1, 0, 1, 1] -> indices=[1, 3, 4, 0, 0], dynamic_len=3

    Args:
        binary_mask: Array of shape (n,) containing 0s and 1s.

    Returns:
        I_padded: Integer array of shape (n,) with indices of 1s packed to the left,
                  right-padded with 0s.
        dynamic_len: Scalar integer, number of 1s in binary_mask.
    """
    n = binary_mask.shape[0]
    indices = jnp.arange(n)
    # Assign a sort key: 1s get their index, 0s get n (so they sort to the end)
    sort_keys = jnp.where(binary_mask, indices, n)
    sorted_order = jnp.argsort(sort_keys, stable=True)
    dynamic_len = jnp.sum(binary_mask).astype(jnp.int32)
    # Mask out the padding positions with 0
    valid = jnp.arange(n) < dynamic_len
    I_padded = jnp.where(valid, sorted_order, -1)
    return I_padded, dynamic_len

if __name__ == "__main__":
    binary_mask = jnp.array([0, 1, 0, 1, 1])
    I_padded, dynamic_len = convert_to_indices(binary_mask)
    dummy = jnp.arange(5)
    x, mask= shift_indices_to_left(dummy, I_padded, dynamic_len)
    print(I_padded)
    print(dynamic_len)
    print(x, mask)