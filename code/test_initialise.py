from initialise import randomly_distribute_item, make_random_chromosome


def test_randomly_distribute_item():
    # Test that the result is valid. Don't care about specific assignments
    num_hampers = 10
    num_items = 7
    result = randomly_distribute_item(num_hampers, num_items)

    assert result.sum() == num_items
    assert result.shape[0] == num_hampers


def test_make_random_chromosome():
    num_hampers = 25
    units = [5, 3, 5, 2, 10]
    result = make_random_chromosome(num_hampers, units)

    assert result.shape == (5, 25)
    assert all(result[i, :].sum() == units[i] for i in range(len(units)))

