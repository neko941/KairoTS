"""Versioned client state store with duplicate operation rejection."""

from kairots.runtime.contracts import ClientState, Result


class StateStore:
    def __init__(self, *, num_clients: int) -> None:
        self.clients = {client_id: ClientState() for client_id in range(num_clients)}
        self.accepted_operations: set[tuple[int, int, str]] = set()

    def accept(self, *, result: Result) -> None:
        if result.operation_id in self.accepted_operations:
            raise ValueError(f"duplicate operation_id rejected: {result.operation_id}")
        current = self.clients[result.operation_id[1]]
        if result.state.version != current.version + 1:
            raise ValueError(
                f"stale client state version for operation_id {result.operation_id}"
            )
        self.accepted_operations.add(result.operation_id)
        self.clients[result.operation_id[1]] = result.state
