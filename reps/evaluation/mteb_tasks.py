from mteb import (
    ArguAna as MTEBArguAna,
    ClimateFEVER as MTEBClimateFEVER,
    CQADupstackAndroidRetrieval as MTEBCQADupstackAndroidRetrieval,
    CQADupstackEnglishRetrieval as MTEBCQADupstackEnglishRetrieval,
    CQADupstackGamingRetrieval as MTEBCQADupstackGamingRetrieval,
    CQADupstackGisRetrieval as MTEBCQADupstackGisRetrieval,
    CQADupstackMathematicaRetrieval as MTEBCQADupstackMathematicaRetrieval,
    CQADupstackPhysicsRetrieval as MTEBCQADupstackPhysicsRetrieval,
    CQADupstackProgrammersRetrieval as MTEBCQADupstackProgrammersRetrieval,
    CQADupstackStatsRetrieval as MTEBCQADupstackStatsRetrieval,
    CQADupstackTexRetrieval as MTEBCQADupstackTexRetrieval,
    CQADupstackUnixRetrieval as MTEBCQADupstackUnixRetrieval,
    CQADupstackWebmastersRetrieval as MTEBCQADupstackWebmastersRetrieval,
    CQADupstackWordpressRetrieval as MTEBCQADupstackWordpressRetrieval,
    DBPedia as MTEBDBPedia,
    FEVER as MTEBFEVER,
    FiQA2018 as MTEBFiQA2018,
    HotpotQA as MTEBHotpotQA,
    MSMARCO as MTEBMSMARCO,
    MSMARCOv2 as MTEBMSMARCOv2,
    NFCorpus as MTEBNFCorpus,
    NQ as MTEBNQ,
    QuoraRetrieval as MTEBQuoraRetrieval,
    SCIDOCS as MTEBSCIDOCS,
    SciFact as MTEBSciFact,
    Touche2020 as MTEBTouche2020,
    TRECCOVID as MTEBTRECCOVID,
    T2Retrieval as MTEBT2Retrieval,
    MMarcoRetrieval as MTEBMMarcoRetrieval,
    DuRetrieval as MTEBDuRetrieval,
    CovidRetrieval as MTEBCovidRetrieval,
    CmedqaRetrieval as MTEBCmedqaRetrieval,
    EcomRetrieval as MTEBEcomRetrieval,
    MedicalRetrieval as MTEBMedicalRetrieval,
    VideoRetrieval as MTEBVideoRetrieval,
    ArguAnaPL as MTEBArguAnaPL,
    DBPediaPL as MTEBDBPediaPL,
    FiQAPLRetrieval as MTEBFiQAPLRetrieval,
    HotpotQAPL as MTEBHotpotQAPL,
    MSMARCOPL as MTEBMSMARCOPL,
    NFCorpusPL as MTEBNFCorpusPL,
    NQPL as MTEBNQPL,
    QuoraPLRetrieval as MTEBQuoraPLRetrieval,
    SCIDOCSPL as MTEBSCIDOCSPL,
    SciFactPL as MTEBSciFactPL,
    TRECCOVIDPL as MTEBTRECCOVIDPL,
)
from reps.evaluation.indexed_task import IndexedTask

class ArguAna(IndexedTask,MTEBArguAna):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class ClimateFEVER(IndexedTask,MTEBClimateFEVER):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class CQADupstackAndroidRetrieval(IndexedTask,MTEBCQADupstackAndroidRetrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class CQADupstackEnglishRetrieval(IndexedTask,MTEBCQADupstackEnglishRetrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class CQADupstackGamingRetrieval(IndexedTask,MTEBCQADupstackGamingRetrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class CQADupstackGisRetrieval(IndexedTask,MTEBCQADupstackGisRetrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class CQADupstackMathematicaRetrieval(IndexedTask,MTEBCQADupstackMathematicaRetrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class CQADupstackPhysicsRetrieval(IndexedTask,MTEBCQADupstackPhysicsRetrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class CQADupstackProgrammersRetrieval(IndexedTask,MTEBCQADupstackProgrammersRetrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class CQADupstackStatsRetrieval(IndexedTask,MTEBCQADupstackStatsRetrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class CQADupstackTexRetrieval(IndexedTask,MTEBCQADupstackTexRetrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class CQADupstackUnixRetrieval(IndexedTask,MTEBCQADupstackUnixRetrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class CQADupstackWebmastersRetrieval(IndexedTask,MTEBCQADupstackWebmastersRetrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class CQADupstackWordpressRetrieval(IndexedTask,MTEBCQADupstackWordpressRetrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class DBPedia(IndexedTask,MTEBDBPedia):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class FEVER(IndexedTask,MTEBFEVER):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class FiQA2018(IndexedTask,MTEBFiQA2018):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class HotpotQA(IndexedTask,MTEBHotpotQA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class MSMARCO(IndexedTask,MTEBMSMARCO):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class MSMARCOv2(IndexedTask,MTEBMSMARCOv2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class NFCorpus(IndexedTask,MTEBNFCorpus):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class NQ(IndexedTask,MTEBNQ):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class QuoraRetrieval(IndexedTask,MTEBQuoraRetrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class SCIDOCS(IndexedTask,MTEBSCIDOCS):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class SciFact(IndexedTask,MTEBSciFact):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class Touche2020(IndexedTask,MTEBTouche2020):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class TRECCOVID(IndexedTask,MTEBTRECCOVID):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class T2Retrieval(IndexedTask,MTEBT2Retrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class MMarcoRetrieval(IndexedTask,MTEBMMarcoRetrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class DuRetrieval(IndexedTask,MTEBDuRetrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class CovidRetrieval(IndexedTask,MTEBCovidRetrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class CmedqaRetrieval(IndexedTask,MTEBCmedqaRetrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class EcomRetrieval(IndexedTask,MTEBEcomRetrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class MedicalRetrieval(IndexedTask,MTEBMedicalRetrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class VideoRetrieval(IndexedTask,MTEBVideoRetrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class ArguAnaPL(IndexedTask,MTEBArguAnaPL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class DBPediaPL(IndexedTask,MTEBDBPediaPL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class FiQAPLRetrieval(IndexedTask,MTEBFiQAPLRetrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class HotpotQAPL(IndexedTask,MTEBHotpotQAPL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class MSMARCOPL(IndexedTask,MTEBMSMARCOPL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class NFCorpusPL(IndexedTask,MTEBNFCorpusPL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class NQPL(IndexedTask,MTEBNQPL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class QuoraPLRetrieval(IndexedTask,MTEBQuoraPLRetrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class SCIDOCSPL(IndexedTask,MTEBSCIDOCSPL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class SciFactPL(IndexedTask,MTEBSciFactPL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class TRECCOVIDPL(IndexedTask,MTEBTRECCOVIDPL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

