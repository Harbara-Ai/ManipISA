from pathlib import Path
import h5py, json, numpy as np
p = Path('teleopdata/dataset/27_ball_box_loading/origin-generalization/episode_000000.hdf5')
with h5py.File(p,'r') as f:
    print('root',list(f))
    print('qpos',f['robot/qpos'].shape)
    for k,v in f['meta'].items():
        if isinstance(v,h5py.Dataset):
            x=v[()]
            if isinstance(x,bytes): x=x.decode()
            if not isinstance(x,np.ndarray) or x.size<10: print(k, str(x)[:8000])
    print('robot',list(f['robot']))
