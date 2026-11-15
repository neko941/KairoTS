from kairots.topologies.FullyConnected import FullyConnected
from kairots.topologies.Ring import Ring


def test_ring_neighbors_are_sorted_and_symmetric() -> None:
    topology = Ring(config=Ring.Config()).neighbors(num_nodes=4)
    assert topology == {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [0, 2]}
    assert Ring(config=Ring.Config()).neighbors(num_nodes=2) == {0: [1], 1: [0]}
    assert Ring(config=Ring.Config()).neighbors(num_nodes=1) == {0: []}
    assert all(
        node in topology[peer] for node, peers in topology.items() for peer in peers
    )


def test_fully_connected_neighbors_are_every_other_node() -> None:
    topology = FullyConnected(config=FullyConnected.Config()).neighbors(num_nodes=3)
    assert topology == {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    assert all(
        node in topology[peer] for node, peers in topology.items() for peer in peers
    )
