from pathlib import Path
import json, sys
import h5py, numpy as np, cv2
ROOT=Path(__file__).resolve().parents[1]
path=Path(sys.argv[1]).resolve()
source=ROOT/'teleopdata/dataset/27_ball_box_loading/origin-generalization/episode_000000.hdf5'
report={'output':str(path),'scope':'Kinematic replay with official RGB and TacMap capture, not learned policy evaluation.'}
with h5py.File(source,'r') as a,h5py.File(path,'r') as f:
    n=int(f['meta/frame_count'][()]); report['frames']=n
    report['qpos_identical_to_source']=bool(np.array_equal(a['robot/qpos'][:],f['robot/qpos'][:]))
    report['object_poses_identical_to_source']=all(np.array_equal(a['objects'][k]['pose_world'][:],f['objects'][k]['pose_world'][:]) for k in a['objects'])
    report['cameras']={}
    for key in f['cameras']:
        g=f['cameras'][key]
        if 'rgb' not in g: continue
        ds=g['rgb']; stats=[]
        for i in [0,n//2,n-1]:
            im=cv2.imdecode(ds[i],cv2.IMREAD_COLOR)
            assert im is not None, (key,i)
            stats.append({'frame':i,'shape':list(im.shape),'std':float(im.std()),'mean':float(im.mean())})
        report['cameras'][key]={'frames':len(ds),'samples':stats}
        assert len(ds)==n and all(x['std']>1 for x in stats),key
    report['tactile']={}
    for key,ds in f['robot/tactile/tacmap'].items():
        arr=ds[:]
        report['tactile'][key]={'shape':list(ds.shape),'nonzero_values':int(np.count_nonzero(arr)),'max':int(arr.max())}
        assert len(ds)==n,key
    assert len(report['cameras'])==6
    assert report['qpos_identical_to_source'] and report['object_poses_identical_to_source']
    cams=['cam_chest','cam_overhead']
    first=[cv2.imdecode(f['cameras'][k]['rgb'][0],cv2.IMREAD_COLOR) for k in cams]
    h=first[0].shape[0]; w=sum(im.shape[1] for im in first)
    out=cv2.VideoWriter(str(path.parent/'preview.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(w,h))
    assert out.isOpened()
    for i in range(n): out.write(np.concatenate([cv2.imdecode(f['cameras'][k]['rgb'][i],cv2.IMREAD_COLOR) for k in cams],axis=1))
    out.release()
report['validation_passed']=True
(path.parent/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'validation_passed':True,'frames':report['frames'],'cameras':len(report['cameras']),'tactile_sites':len(report['tactile']),'video':str(path.parent/'preview.mp4')},indent=2))

