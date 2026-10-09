"""Искусственные записи завершённых путей для анализа пар."""
import json
from pathlib import Path
import tempfile
from experiments.source_recovery.analysis import summary as s

class CampaignFixture:
    """Искусственная кампания с принятой, исключённой и отсутствующей парой."""

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.condition=dict(id="main",sources=["PG10"],replicates=[1,2,3],penalties=["L2","H1"],
            weight="W03",noise="corr",availability="full",mask="none",nodes=73,grid="G0",
            tau_hours=.25,temporal_H="snapshot",relocation_km=0,truth_gamma=1/72,inverse_gamma=1/72)
        spec=dict(sources=["PG10"],replicates=[1,2,3],conditions=[self.condition],single_fits=[])
        self.write("run.json",dict(configuration=spec,bindings={"config":"frozen"}))
        for r in (1,2):
            paths={}
            scores={}
            for arm,value in (("L2",.3),("H1",.2)):
                pid="main/"+arm
                ok=not(r==2 and arm=="H1")
                paths[pid]=dict(finalized=True,complete_path=True,procedure_accepted=ok,tuning_unresolved=not ok,
                    final_certificate={"accepted":True},candidates={str(i):{"accepted":True,"status":"accepted"} for i in range(25)})
                scores[pid]=dict(E_q=value,relative_L2=value*2,full_primary_test_rmse=value*3,
                    full_primary_noiseless_rmse=value,score_row_count=36,diagnostic_only=not ok)
            cp=dict(bindings={"config":"frozen"},expected_paths=list(paths),paths=paths,scores=scores,
                    exponents=[-8+i*.5 for i in range(25)],stage="scored",selection_seal=s.selection_hash(paths))
            self.write(f"PG10/replicate_{r}.json",cp)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self,path,payload):
        p=self.root/path
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(payload),encoding="utf-8")
