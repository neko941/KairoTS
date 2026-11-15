"""SMD machines 1-1 to 1-8 (group 1), one client each."""

from dataclasses import dataclass

from kairots.data.sources.SMD import SMD


class SMDGroup1(SMD):
    TASKS = ("AnomalyDetection",)
    MACHINES = {
        "machine-1-1": (
            "ecebd2933ccdc1126e2abbfb538762dbde16410cebc77257a7b96e1e1fbcf753",
            "bdb801d8d52ce6b0c4a20c310edf31b5cee9bc82eb84fb38e7d4a19309b16ddb",
            "cddf33441ae480c6074e173c897d4715eb893f24a7078d3d3e1b79b200c7cd19",
        ),
        "machine-1-2": (
            "7f6f63ec34ce616c87b4c30d66fd52ad1b21149634ae3ff7bb1827a76d6ef79a",
            "f409991af7c049d3f286a8cf89dc2fa99114bea8753f6c3aea4bd24cb03eb015",
            "c2dbb3e4f89c9eb48f30fd1ddae48b80e7ae7ad7ca9abf2ffd71b6557ef2bb20",
        ),
        "machine-1-3": (
            "a4ef16a49d921e0ba9497ce2690d3833eb4a85f9a8a88392de4c3fde42e413ac",
            "f3c134d45ffa7aed66587dba893111396c8c7ea3b6fa7f76895955b26b1a75ef",
            "755d57179625682172ae85f9483749cff576ff3282d0af294271a40ab6ea7b28",
        ),
        "machine-1-4": (
            "2ec56c43f91684aa4751a6b72d359669266712dd69bdc7efda3dc60c0ade1156",
            "98ade00e57eac63e8f81a506a8e848803b58590835ab2b3cf53d1f0459900bd3",
            "a470a3f7b66ec3d080fc5ee3d9565e0f3bfcfa16a1650259f771736327fa9a8b",
        ),
        "machine-1-5": (
            "69061055f43413863434cb9c0424c14b13a7a0817056a8fd9439e53dc365fc0b",
            "98b8979f21b4b68c7308f38dd093b9bff4275814578365103c692f28f60f61c5",
            "e7e2e611e12b5e9e2558ed955ad271f1f408fbdd7fbbfdfc72eb745ade567815",
        ),
        "machine-1-6": (
            "27e1765c04c2a5d08f39d98e7979e7764886fc733c36382fe081157a5ac0b9d8",
            "0797fe4bfd25ef13c830955cb9f2a36b4c9d5da265b8e91702408b6b048267f4",
            "8359ffa67b5421b0b9967ccf396a58c7b3bf52fa24c3f95e6b9df6486f1d6365",
        ),
        "machine-1-7": (
            "dddb9f770bdb8b1b952263e0cc4f000b6f8ec612941dd3eb7a303994f257edf8",
            "785a3f542ef6ac171a3af55c534f9ca598d53fa0371010172c72e33af4588d16",
            "24b3ce00bf2c56b571775a2b28d4e50ed27d050d77a031fa16c11ef3e51809b8",
        ),
        "machine-1-8": (
            "3b46e9754ec06bebd0bf3fe68dbe28b2a4b67298e795c9501d97f42083e337fc",
            "b8784c3b4169f8a0a2ed9f47b00070d72ed214dae0307553b8cfdb80c712d779",
            "8019476032c4c5b4b80127cfb9e74ee7d57fbbd558aa15097518713e16463a47",
        ),
    }

    @dataclass(kw_only=True)
    class Config(SMD.Config):
        name: str = "SMDGroup1"
