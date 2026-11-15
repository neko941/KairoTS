"""TourismLarge: Australian domestic tourism, as Nixtla's hierarchical datasets.

Monthly visitor nights from 1998-01 to 2016-12 (228 steps) in 555 series: the
total, 7 states, 27 zones and 76 regions, each in total (`All`) and by travel
purpose (`Hol`, `Vis`, `Bus`, `Oth`). The 304 region-by-purpose series are the
bottom level; every other series is a sum of them. One file is one client.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from kairots.data.download import RemoteFile
from kairots.data.sources.Hierarchical import Hierarchical

CHANNELS = tuple(
    (
        "TotalAll AAll BAll CAll DAll EAll FAll GAll AAAll ABAll ACAll ADAll AEAll "
        "AFAll BAAll BBAll BCAll BDAll BEAll CAAll CBAll CCAll CDAll DAAll DBAll "
        "DCAll DDAll EAAll EBAll ECAll FAAll FBAll FCAll GAAll GBAll AAAAll AABAll "
        "ABAAll ABBAll ACAAll ADAAll ADBAll ADCAll ADDAll AEAAll AEBAll AECAll AEDAll "
        "AFAAll BAAAll BABAll BACAll BBAAll BCAAll BCBAll BCCAll BDAAll BDBAll BDCAll "
        "BDDAll BDEAll BDFAll BEAAll BEBAll BECAll BEDAll BEEAll BEFAll BEGAll BEHAll "
        "CAAAll CABAll CACAll CBAAll CBBAll CBCAll CBDAll CCAAll CCBAll CCCAll CDAAll "
        "CDBAll DAAAll DABAll DACAll DBAAll DBBAll DBCAll DCAAll DCBAll DCCAll DCDAll "
        "DDAAll DDBAll EAAAll EABAll EACAll EBAAll ECAAll FAAAll FBAAll FBBAll FCAAll "
        "FCBAll GAAAll GABAll GACAll GBAAll GBBAll GBCAll GBDAll TotalHol TotalVis "
        "TotalBus TotalOth AHol AVis ABus AOth BHol BVis BBus BOth CHol CVis CBus "
        "COth DHol DVis DBus DOth EHol EVis EBus EOth FHol FVis FBus FOth GHol GVis "
        "GBus GOth AAHol AAVis AABus AAOth ABHol ABVis ABBus ABOth ACHol ACVis ACBus "
        "ACOth ADHol ADVis ADBus ADOth AEHol AEVis AEBus AEOth AFHol AFVis AFBus "
        "AFOth BAHol BAVis BABus BAOth BBHol BBVis BBBus BBOth BCHol BCVis BCBus "
        "BCOth BDHol BDVis BDBus BDOth BEHol BEVis BEBus BEOth CAHol CAVis CABus "
        "CAOth CBHol CBVis CBBus CBOth CCHol CCVis CCBus CCOth CDHol CDVis CDBus "
        "CDOth DAHol DAVis DABus DAOth DBHol DBVis DBBus DBOth DCHol DCVis DCBus "
        "DCOth DDHol DDVis DDBus DDOth EAHol EAVis EABus EAOth EBHol EBVis EBBus "
        "EBOth ECHol ECVis ECBus ECOth FAHol FAVis FABus FAOth FBHol FBVis FBBus "
        "FBOth FCHol FCVis FCBus FCOth GAHol GAVis GABus GAOth GBHol GBVis GBBus "
        "GBOth AAAHol AAAVis AAABus AAAOth AABHol AABVis AABBus AABOth ABAHol ABAVis "
        "ABABus ABAOth ABBHol ABBVis ABBBus ABBOth ACAHol ACAVis ACABus ACAOth ADAHol "
        "ADAVis ADABus ADAOth ADBHol ADBVis ADBBus ADBOth ADCHol ADCVis ADCBus ADCOth "
        "ADDHol ADDVis ADDBus ADDOth AEAHol AEAVis AEABus AEAOth AEBHol AEBVis AEBBus "
        "AEBOth AECHol AECVis AECBus AECOth AEDHol AEDVis AEDBus AEDOth AFAHol AFAVis "
        "AFABus AFAOth BAAHol BAAVis BAABus BAAOth BABHol BABVis BABBus BABOth BACHol "
        "BACVis BACBus BACOth BBAHol BBAVis BBABus BBAOth BCAHol BCAVis BCABus BCAOth "
        "BCBHol BCBVis BCBBus BCBOth BCCHol BCCVis BCCBus BCCOth BDAHol BDAVis BDABus "
        "BDAOth BDBHol BDBVis BDBBus BDBOth BDCHol BDCVis BDCBus BDCOth BDDHol BDDVis "
        "BDDBus BDDOth BDEHol BDEVis BDEBus BDEOth BDFHol BDFVis BDFBus BDFOth BEAHol "
        "BEAVis BEABus BEAOth BEBHol BEBVis BEBBus BEBOth BECHol BECVis BECBus BECOth "
        "BEDHol BEDVis BEDBus BEDOth BEEHol BEEVis BEEBus BEEOth BEFHol BEFVis BEFBus "
        "BEFOth BEGHol BEGVis BEGBus BEGOth BEHHol BEHVis BEHBus BEHOth CAAHol CAAVis "
        "CAABus CAAOth CABHol CABVis CABBus CABOth CACHol CACVis CACBus CACOth CBAHol "
        "CBAVis CBABus CBAOth CBBHol CBBVis CBBBus CBBOth CBCHol CBCVis CBCBus CBCOth "
        "CBDHol CBDVis CBDBus CBDOth CCAHol CCAVis CCABus CCAOth CCBHol CCBVis CCBBus "
        "CCBOth CCCHol CCCVis CCCBus CCCOth CDAHol CDAVis CDABus CDAOth CDBHol CDBVis "
        "CDBBus CDBOth DAAHol DAAVis DAABus DAAOth DABHol DABVis DABBus DABOth DACHol "
        "DACVis DACBus DACOth DBAHol DBAVis DBABus DBAOth DBBHol DBBVis DBBBus DBBOth "
        "DBCHol DBCVis DBCBus DBCOth DCAHol DCAVis DCABus DCAOth DCBHol DCBVis DCBBus "
        "DCBOth DCCHol DCCVis DCCBus DCCOth DCDHol DCDVis DCDBus DCDOth DDAHol DDAVis "
        "DDABus DDAOth DDBHol DDBVis DDBBus DDBOth EAAHol EAAVis EAABus EAAOth EABHol "
        "EABVis EABBus EABOth EACHol EACVis EACBus EACOth EBAHol EBAVis EBABus EBAOth "
        "ECAHol ECAVis ECABus ECAOth FAAHol FAAVis FAABus FAAOth FBAHol FBAVis FBABus "
        "FBAOth FBBHol FBBVis FBBBus FBBOth FCAHol FCAVis FCABus FCAOth FCBHol FCBVis "
        "FCBBus FCBOth GAAHol GAAVis GAABus GAAOth GABHol GABVis GABBus GABOth GACHol "
        "GACVis GACBus GACOth GBAHol GBAVis GBABus GBAOth GBBHol GBBVis GBBBus GBBOth "
        "GBCHol GBCVis GBCBus GBCOth GBDHol GBDVis GBDBus GBDOth "
    ).split()
)


@dataclass(kw_only=True)
class TourismLarge(Hierarchical):
    TASKS = ("Forecasting",)
    URL = "https://nixtla-public.s3.amazonaws.com/hierarchical-data/datasets.zip"
    SHA256 = "6512d9aa80f111ee26480bc6f3f4eb3b5655d4ceecc384933100edc85adf704b"

    @dataclass(kw_only=True)
    class Config(Hierarchical.Config):
        name: str = "TourismLarge"
        raw_path: str = "datasets/TourismLarge/raw"
        input_channels: tuple[str, ...] = CHANNELS
        seasonality: int = 12

    @classmethod
    def raw_files(cls, *, config: Config) -> list[Path]:
        """Raw files this source reads, known without reading them."""
        root = Path(config.raw_path) / "hierarchical" / "TourismLarge"
        return [root / "data.csv", root / "agg_mat.csv"]

    @classmethod
    def remote_files(cls, *, config: Config) -> tuple[RemoteFile, ...]:
        """Nixtla's archive of hierarchical datasets."""
        return (
            RemoteFile(
                path="datasets.zip",
                urls=(cls.URL,),
                sha256=cls.SHA256,
                extract="hierarchical",
            ),
        )

    @classmethod
    def load(cls, *, config: Config) -> TourismLarge:
        data_path, matrix_path = cls.raw_files(config=config)
        frame = pl.read_csv(data_path)
        frame = frame.rename({frame.columns[0]: "date"}).with_columns(
            pl.col("date").str.to_datetime("%Y-%m-%d")
        )
        summing = cls.summing(frame=frame, matrix=pl.read_csv(matrix_path))
        missing = set(config.input_channels) - set(frame.columns)
        if missing:
            raise ValueError(f"{data_path.name} missing columns: {sorted(missing)}")
        return cls(
            name=cls.__name__,
            input_channels=config.input_channels,
            timestamps={cls.__name__: tuple(frame.get_column("date").to_list())},
            clients=[(cls.__name__, frame.select(["date", *config.input_channels]))],
            manifest_fields={
                "raw_files": [str(data_path), str(matrix_path)],
                "frequency": "monthly",
                "bottom": [
                    name for name in config.input_channels if summing[name] == [name]
                ],
                "summing": {name: summing[name] for name in config.input_channels},
            },
        )
