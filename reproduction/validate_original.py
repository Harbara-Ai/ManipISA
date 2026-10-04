from pathlib import Path
import hashlib, json, sys
import h5py, numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Bench2Dex'))
from tools.batch_re_eval import load_states_from_hdf5, _resolve_eval_frame
from tools.label_success import label_episode
from success.custom.task_27_ball_box_loading import check_success
source=ROOT/'teleopdata/dataset/27_ball_box_loading/origin-generalization/episode_000000.hdf5'
with h5py.File(source,'r') as f:
    def val(k):
        x=f['meta/'+k][()]
        return x.decode() if isinstance(x,bytes) else x.item() if isinstance(x,np.generic) else x
    report={k:val(k) for k in ['frame_count','robot_key','action_dim','action_type','effective_fps','step_stride','homing_start_sim_step','success','collection_mode','demo_eligible']}
    report['moving_joint_count']=int((np.ptp(f['robot/qpos'][:],axis=0)>0.01).sum())
    report['joint_names_unique']=len(set(f['robot/joint_names'][:]))==52
    report['qpos_finite']=bool(np.isfinite(f['robot/qpos'][:]).all())
report['source_sha256']=hashlib.file_digest(source.open('rb'),'sha256').hexdigest()
report['upstream_eval_frame']=_resolve_eval_frame(str(source))
report['upstream_success_at_eval_frame']=bool(check_success(load_states_from_hdf5(str(source)),{},{}))
report['upstream_stable_success_recomputed']=label_episode(str(source),check_success,force=True,dry_run=True)
report['metadata_caveat']='Upstream collector/data_collector.py hard-codes collection_mode=scripted_hold and demo_eligible=False (lines 701-723); these fields alone cannot establish collection provenance or training eligibility.'
report['scope']='Official recorded trajectory validation; not a policy rollout or dynamic control success rate.'
(ROOT/'reproduction/original-validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))

