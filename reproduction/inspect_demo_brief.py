import json, h5py, numpy as np
with h5py.File('teleopdata/dataset/27_ball_box_loading/origin-generalization/episode_000000.hdf5','r') as f:
    for k in ['robot_key','robot_name','robot_config','provenance','collection_mode','demo_eligible','num_frames','frame_count','success','homing_start_sim_step']:
        if k in f['meta']: print(k, f['meta'][k][()])
    q=f['robot/qpos'][:]; print('joint motion: range max',np.ptp(q,axis=0).max(),'moving joints',sum(np.ptp(q,axis=0)>0.01))
    print('time range',f['time/sim_step'][0],f['time/sim_step'][-1])
    for k in f['meta']:
        if 'generalization' in k:
            x=f['meta'][k][()]
            if isinstance(x,bytes):
                try:
                    j=json.loads(x)
                    def walk(x):
                        if isinstance(x,dict):
                            for k,v in x.items():
                                if isinstance(v,str) and ('dex2bench_dataset/' in v): print(v)
                                else: walk(v)
                        elif isinstance(x,list):
                            for y in x: walk(y)
                    walk(j)
                except Exception: pass
