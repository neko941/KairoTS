"""Select reusable client and phase populations for evaluation protocols."""


def select(
    *,
    client_ids: list[int],
    unseen_client_ids: set[int],
    phases: list[str],
    cross_dataset: bool,
) -> dict[str, tuple[list[int], list[str]]]:
    if not unseen_client_ids.issubset(client_ids):
        raise ValueError("unseen clients must be included in the loaded population")
    return {
        "cross_" if cross_dataset else "": (
            [
                client_id
                for client_id in client_ids
                if client_id not in unseen_client_ids
            ],
            phases,
        ),
        "unseen_": (
            [client_id for client_id in client_ids if client_id in unseen_client_ids],
            ["test"],
        ),
    }
