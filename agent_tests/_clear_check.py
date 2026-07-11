import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from dial_mpc.costs import make_grasp_cost, make_rekep_cost
from sim_common.constraints import TorchNumpyShim, make_torch_constraint
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
class FK:
    def grasp_point(self, q, off):
        b=q.shape[0]; return q[:,:3].contiguous(), torch.eye(3,device=q.device).expand(b,3,3)
k,h=16,8
qt=torch.randn(k,h,7,device=DEV); qc=torch.zeros(7,device=DEV)
obs=[[0.1,0.0,0.2],[0.3,0.1,0.2]]
g=make_grasp_cost(FK(),[0,0,1],w_clear=50.0,obstacles=obs,obstacle_r=0.04,device=DEV)
o=g(qt,qc,(torch.tensor([0.2,0.,0.2],device=DEV),)*2)
print(f"[clear] grasp+obstacles: shape={tuple(o.shape)} finite={bool(torch.isfinite(o).all())}")
assert o.shape==(k,) and torch.isfinite(o).all()
# obstacles=None path unchanged
g0=make_grasp_cost(FK(),[0,0,1],device=DEV)
assert g0(qt,qc,(torch.tensor([0.2,0.,0.2],device=DEV),)*2).shape==(k,)
sh=TorchNumpyShim(DEV); src="def f(end_effector,keypoints):\n return np.linalg.norm(keypoints[0]-keypoints[1])\n"
gl={"np":sh,"get_grasping_cost_by_keypoint_idx":lambda *a:0.0}; lv={}; exec(src,gl,lv)
cf=make_torch_constraint(list(lv.values()))
r=make_rekep_cost(FK(),cf,[0,0,1],held_idx=[0],held_offset=[[0,0,0]],w_clear=50.0,obstacles=obs,device=DEV)
oo=r(qt,qc,torch.randn(5,3,device=DEV))
print(f"[clear] rekep+held+obstacles: shape={tuple(oo.shape)} finite={bool(torch.isfinite(oo).all())}")
assert oo.shape==(k,) and torch.isfinite(oo).all()
print("[clear] OK")
