"""
Debug the optimization issue
"""

import torch


# Test the dilation logic
def test_dilation_equivalence():
    print("Testing Dilation Equivalence")
    print("=" * 40)

    # Create test tensor
    x = torch.arange(16).reshape(1, 1, 16, 1, 1).float()
    print(f"Input: {x.squeeze().tolist()}")

    # Test with dilation_rate=2, offset=0
    r = 2
    offset = 0

    # Method 1: index_select (original)
    idx = torch.arange(offset, 16, r)
    result1 = x.index_select(2, idx)
    print(f"\nindex_select result: {result1.squeeze().tolist()}")

    # Method 2: direct slicing
    result2 = x[:, :, ::r, :, :]
    print(f"Direct slice result: {result2.squeeze().tolist()}")

    print(f"Results match: {torch.allclose(result1, result2)}")

    # Test with offset=1
    print("\n" + "-" * 40)
    offset = 1
    idx = torch.arange(offset, 16, r)
    result3 = x.index_select(2, idx)
    print(f"\nindex_select (offset=1): {result3.squeeze().tolist()}")

    # Advanced indexing
    result4 = x[:, :, idx, :, :]
    print(f"Advanced indexing result: {result4.squeeze().tolist()}")

    print(f"Results match: {torch.allclose(result3, result4)}")


def test_dilation_rate_one():
    """Test special case where dilation_rate=1"""
    print("\n\nTesting Dilation Rate = 1")
    print("=" * 40)

    x = torch.arange(8).reshape(1, 1, 8, 1, 1).float()
    r = 1
    offset = 0  # offset = i % r = 0 % 1 = 0

    print(f"Input: {x.squeeze().tolist()}")
    print(f"Dilation rate: {r}, offset: {offset}")

    # The code checks if r > 1, so with r=1 no dilation is applied
    print("With r=1, no dilation should be applied")
    print("This is handled by the if r > 1 check")


test_dilation_equivalence()
test_dilation_rate_one()
