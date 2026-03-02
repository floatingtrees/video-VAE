import jax
import jax.numpy as jnp
from shift_indices import shift_indices_to_left, convert_to_indices


def test_basic():
    V = jnp.array([10, 20, 30, 40, 50])
    I_padded = jnp.array([3, 1, 4, 0, 0])
    dynamic_len = jnp.array(3)

    T, mask = shift_indices_to_left(V, I_padded, dynamic_len)
    assert jnp.array_equal(T, jnp.array([40, 20, 50, 0, 0])), f"Got {T}"
    assert jnp.array_equal(mask, jnp.array([True, True, True, False, False])), f"Got {mask}"
    print("test_basic passed")


def test_all_valid():
    V = jnp.array([10, 20, 30])
    I_padded = jnp.array([2, 0, 1])
    dynamic_len = jnp.array(3)

    T, mask = shift_indices_to_left(V, I_padded, dynamic_len)
    assert jnp.array_equal(T, jnp.array([30, 10, 20])), f"Got {T}"
    assert jnp.all(mask), f"Got {mask}"
    print("test_all_valid passed")


def test_none_valid():
    V = jnp.array([10, 20, 30])
    I_padded = jnp.array([0, 0, 0])
    dynamic_len = jnp.array(0)

    T, mask = shift_indices_to_left(V, I_padded, dynamic_len)
    assert jnp.array_equal(T, jnp.array([0, 0, 0])), f"Got {T}"
    assert not jnp.any(mask), f"Got {mask}"
    print("test_none_valid passed")


def test_one_valid():
    V = jnp.array([10, 20, 30, 40])
    I_padded = jnp.array([2, 0, 0, 0])
    dynamic_len = jnp.array(1)

    T, mask = shift_indices_to_left(V, I_padded, dynamic_len)
    assert jnp.array_equal(T, jnp.array([30, 0, 0, 0])), f"Got {T}"
    assert jnp.array_equal(mask, jnp.array([True, False, False, False])), f"Got {mask}"
    print("test_one_valid passed")


def test_multidim():
    """Test with V of shape (n, d) - e.g. feature vectors."""
    V = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]])  # (4, 2)
    I_padded = jnp.array([3, 1, 0, 0])
    dynamic_len = jnp.array(2)

    T, mask = shift_indices_to_left(V, I_padded, dynamic_len)
    expected = jnp.array([[7, 8], [3, 4], [0, 0], [0, 0]])
    assert jnp.array_equal(T, expected), f"Got {T}"
    assert jnp.array_equal(mask, jnp.array([True, True, False, False])), f"Got {mask}"
    print("test_multidim passed")


def test_jit_compatible():
    """Verify the function works under jax.jit."""
    V = jnp.array([10, 20, 30, 40, 50])
    I_padded = jnp.array([3, 1, 4, 0, 0])
    dynamic_len = jnp.array(3)

    jit_fn = jax.jit(shift_indices_to_left)
    T, mask = jit_fn(V, I_padded, dynamic_len)
    assert jnp.array_equal(T, jnp.array([40, 20, 50, 0, 0])), f"Got {T}"
    assert jnp.array_equal(mask, jnp.array([True, True, True, False, False])), f"Got {mask}"
    print("test_jit_compatible passed")


def test_jit_different_dynamic_lens():
    """Verify JIT works with different dynamic_len values without retracing."""
    V = jnp.array([10, 20, 30, 40, 50])
    jit_fn = jax.jit(shift_indices_to_left)

    T1, m1 = jit_fn(V, jnp.array([3, 1, 4, 0, 0]), jnp.array(3))
    assert jnp.array_equal(T1, jnp.array([40, 20, 50, 0, 0])), f"Got {T1}"

    T2, m2 = jit_fn(V, jnp.array([0, 2, 0, 0, 0]), jnp.array(2))
    assert jnp.array_equal(T2, jnp.array([10, 30, 0, 0, 0])), f"Got {T2}"

    T3, m3 = jit_fn(V, jnp.array([4, 3, 2, 1, 0]), jnp.array(5))
    assert jnp.array_equal(T3, jnp.array([50, 40, 30, 20, 10])), f"Got {T3}"
    print("test_jit_different_dynamic_lens passed")


def test_convert_basic():
    mask = jnp.array([0, 1, 0, 1, 1])
    I_padded, dynamic_len = convert_to_indices(mask)
    assert int(dynamic_len) == 3, f"Got {dynamic_len}"
    assert jnp.array_equal(I_padded, jnp.array([1, 3, 4, 0, 0])), f"Got {I_padded}"
    print("test_convert_basic passed")


def test_convert_all_ones():
    mask = jnp.array([1, 1, 1])
    I_padded, dynamic_len = convert_to_indices(mask)
    assert int(dynamic_len) == 3, f"Got {dynamic_len}"
    assert jnp.array_equal(I_padded, jnp.array([0, 1, 2])), f"Got {I_padded}"
    print("test_convert_all_ones passed")


def test_convert_all_zeros():
    mask = jnp.array([0, 0, 0, 0])
    I_padded, dynamic_len = convert_to_indices(mask)
    assert int(dynamic_len) == 0, f"Got {dynamic_len}"
    assert jnp.array_equal(I_padded, jnp.array([0, 0, 0, 0])), f"Got {I_padded}"
    print("test_convert_all_zeros passed")


def test_convert_single_one():
    mask = jnp.array([0, 0, 1, 0])
    I_padded, dynamic_len = convert_to_indices(mask)
    assert int(dynamic_len) == 1, f"Got {dynamic_len}"
    assert jnp.array_equal(I_padded, jnp.array([2, 0, 0, 0])), f"Got {I_padded}"
    print("test_convert_single_one passed")


def test_convert_jit():
    mask = jnp.array([1, 0, 1, 0, 1])
    jit_fn = jax.jit(convert_to_indices)
    I_padded, dynamic_len = jit_fn(mask)
    assert int(dynamic_len) == 3, f"Got {dynamic_len}"
    assert jnp.array_equal(I_padded, jnp.array([0, 2, 4, 0, 0])), f"Got {I_padded}"
    print("test_convert_jit passed")


def test_convert_roundtrip():
    """Test that convert_to_indices + shift_indices_to_left correctly gathers selected elements."""
    V = jnp.array([10, 20, 30, 40, 50])
    mask = jnp.array([0, 1, 0, 1, 1])
    I_padded, dynamic_len = convert_to_indices(mask)
    T, valid_mask = shift_indices_to_left(V, I_padded, dynamic_len)
    assert jnp.array_equal(T, jnp.array([20, 40, 50, 0, 0])), f"Got {T}"
    assert jnp.array_equal(valid_mask, jnp.array([True, True, True, False, False])), f"Got {valid_mask}"
    print("test_convert_roundtrip passed")


if __name__ == "__main__":
    test_basic()
    test_all_valid()
    test_none_valid()
    test_one_valid()
    test_multidim()
    test_jit_compatible()
    test_jit_different_dynamic_lens()
    test_convert_basic()
    test_convert_all_ones()
    test_convert_all_zeros()
    test_convert_single_one()
    test_convert_jit()
    test_convert_roundtrip()
    print("\nAll tests passed!")
