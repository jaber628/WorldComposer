from __future__ import annotations
import argparse, json, math, os, sys
from dataclasses import asdict, dataclass
from pathlib import Path
import cv2, numpy as np, open3d as o3d, torch
from PIL import Image
from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[4]
THIRD_PARTY_DIR = PROJECT_ROOT / '3rd'
LANG_SAM_REPO_DIR = THIRD_PARTY_DIR / 'lang-segment-anything'
CACHE_ROOT = THIRD_PARTY_DIR / 'cache'
HF_CACHE = CACHE_ROOT / 'huggingface'
TORCH_CACHE = CACHE_ROOT / 'torch'
XDG_CACHE = CACHE_ROOT / 'xdg'

for cache_dir in (CACHE_ROOT, HF_CACHE, TORCH_CACHE, XDG_CACHE):
    cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('HF_HOME', str(HF_CACHE))
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', str(HF_CACHE / 'hub'))
os.environ.setdefault('TRANSFORMERS_CACHE', str(HF_CACHE / 'transformers'))
os.environ.setdefault('TORCH_HOME', str(TORCH_CACHE))
os.environ.setdefault('XDG_CACHE_HOME', str(XDG_CACHE))
if LANG_SAM_REPO_DIR.exists() and str(LANG_SAM_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(LANG_SAM_REPO_DIR))

parser=argparse.ArgumentParser(description='Real2Sim phase 2: detect tabletop points from multiview renders')
parser.add_argument('--scene_usd',type=str,required=True)
parser.add_argument('--output_dir',type=str,required=True)
parser.add_argument('--pitch_deg',type=float,default=-30.0)
parser.add_argument('--yaw_step_deg',type=float,default=30.0)
parser.add_argument('--camera_origin',type=float,nargs=3,default=(0.0,0.0,0.0))
parser.add_argument('--image_width',type=int,default=960)
parser.add_argument('--image_height',type=int,default=720)
parser.add_argument('--sam_prompt',type=str,default='tabletop')
parser.add_argument('--depth_debug',action='store_true',default=False)
parser.add_argument('--save_debug_images',action='store_true',default=False,help='Save full per-view RGB/depth/mask/overlay debug artifacts')
parser.add_argument('--validate_mesh_depth_only',action='store_true',default=False)
parser.add_argument('--top_k_views',type=int,default=4)
parser.add_argument('--min_view_score',type=float,default=0.35)
parser.add_argument('--refine_yaw_step_deg',type=float,default=5.0)
parser.add_argument('--refine_margin_deg',type=float,default=10.0)
parser.add_argument('--disable_refine_views',action='store_true',default=True)
parser.add_argument('--max_points_per_view',type=int,default=12000)
parser.add_argument('--cluster_eps',type=float,default=0.05)
parser.add_argument('--cluster_min_points',type=int,default=80)
parser.add_argument('--voxel_size',type=float,default=0.01)
AppLauncher.add_app_launcher_args(parser)
if '--enable_cameras' not in sys.argv: sys.argv.append('--enable_cameras')
args=parser.parse_args()
app=AppLauncher(vars(args)).app

import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import matrix_from_quat


def _get_settings():
    try:
        import carb  # type: ignore
        return carb.settings.get_settings()
    except Exception:
        return None


def _safe_set_setting(settings, path, value):
    if settings is None:
        return False
    try:
        settings.set(path, value)
        return True
    except Exception:
        return False


def apply_default_render_settings(*, task_name: str | None = None):
    settings = _get_settings()
    _safe_set_setting(settings, '/rtx/post/tonemap/enabled', True)
    _safe_set_setting(settings, '/rtx/post/tonemap/fNumber', 5.8)
    try:
        import omni.usd  # type: ignore
        from pxr import Gf, Sdf, UsdGeom, UsdLux  # type: ignore
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        for name, intensity, rot in [
            ('Key', 3500.0, (65.0, 25.0, 0.0)),
            ('Fill', 900.0, (80.0, -55.0, 0.0)),
            ('Rim', 600.0, (160.0, 160.0, 0.0)),
        ]:
            path = f'/World/DefaultLight/{name}'
            prim = stage.GetPrimAtPath(path)
            light = UsdLux.DistantLight.Define(stage, path) if not prim.IsValid() else UsdLux.DistantLight(prim)
            light.GetIntensityAttr().Set(float(intensity))
            light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
            xform = UsdGeom.Xformable(light.GetPrim())
            xform.ClearXformOpOrder()
            xform.AddRotateXYZOp().Set(Gf.Vec3f(*rot))
    except Exception:
        pass


def set_scene_visibility(*, show_mesh: bool, show_gauss: bool):
    try:
        import omni.usd  # type: ignore
        from pxr import Sdf, UsdGeom  # type: ignore
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        def _get_first_valid_prim(*paths: str):
            for path in paths:
                prim = stage.GetPrimAtPath(path)
                if prim.IsValid():
                    return prim
            return None

        mesh_prim = _get_first_valid_prim('/World/Scene/Xform', '/World/Scene/World/Xform')
        if mesh_prim is not None:
            imageable = UsdGeom.Imageable(mesh_prim)
            if show_mesh:
                imageable.MakeVisible()
                visibility_attr = imageable.GetVisibilityAttr()
                if visibility_attr.IsValid():
                    visibility_attr.Set(UsdGeom.Tokens.inherited)
                purpose_attr = mesh_prim.GetAttribute('purpose')
                if purpose_attr.IsValid():
                    purpose_attr.Set(UsdGeom.Tokens.default_)
                else:
                    mesh_prim.CreateAttribute('purpose', Sdf.ValueTypeNames.Token).Set(UsdGeom.Tokens.default_)
            else:
                imageable.MakeInvisible()

        gauss_prim = _get_first_valid_prim('/World/Scene/gauss', '/World/Scene/World/gauss')
        if gauss_prim is not None:
            gauss_imageable = UsdGeom.Imageable(gauss_prim)
            if show_gauss:
                gauss_imageable.MakeVisible()
            else:
                gauss_imageable.MakeInvisible()
    except Exception:
        pass

@dataclass
class ViewScore:
    yaw_deg: float; area_score: float; conn_score: float; pos_score: float; smooth_score: float; depth_score: float; final_score: float; area_ratio: float; kept: bool=False; regularity_score: float=0.0; top_bias_score: float=0.0; leg_penalty: float=0.0; fragmentation_penalty: float=0.0
@dataclass
class ClusterStat:
    cluster_id: int; num_points: int; z_median: float; z_std: float; xy_area: float; up_alignment: float; final_score: float

class LangSAMDetector:
    def __init__(self):
        try:
            from lang_sam import LangSAM  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "lang_sam is not installed. Place the LangSAM repository at "
                "'3rd/lang-segment-anything' and install it with "
                "`pip install -e 3rd/lang-segment-anything`.\n"
                f"Expected repository path: {LANG_SAM_REPO_DIR}"
            ) from exc
        self.model=LangSAM()
    def predict(self,rgb:np.ndarray,prompt:str)->list[np.ndarray]:
        img=Image.fromarray(rgb)
        errors=[]

        try:
            res=self.model.predict([img],[prompt])
            return self._extract_masks(res)
        except Exception as exc:
            errors.append(f"predict([image],[prompt]) failed: {exc}")

        for kw in ({'text_prompt':prompt},{'prompt':prompt},{}):
            try:
                res=self.model.predict(img,**kw) if kw else self.model.predict(img)
                return self._extract_masks(res)
            except Exception as exc:
                errors.append(f"predict(image, {kw}) failed: {exc}")

        raise RuntimeError('LangSAM predict API failed. Attempts: ' + ' | '.join(errors))

    @staticmethod
    def _extract_masks(res)->list[np.ndarray]:
        if res is None:
            return []
        if isinstance(res,list):
            masks=[]
            for item in res:
                if isinstance(item,dict) and 'masks' in item:
                    arr=np.asarray(item['masks'])
                    if arr.size==0:
                        continue
                    if arr.ndim==2:
                        masks.append(arr.astype(bool))
                    else:
                        for m in arr:
                            masks.append(np.squeeze(np.asarray(m)).astype(bool))
            return masks
        if isinstance(res,tuple):
            first=res[0]
            if isinstance(first,(list,tuple,np.ndarray)):
                return [np.squeeze(np.asarray(m)).astype(bool) for m in first]
        if isinstance(res,dict) and 'masks' in res:
            arr=np.asarray(res['masks'])
            if arr.ndim==2:
                return [arr.astype(bool)]
            return [np.squeeze(np.asarray(m)).astype(bool) for m in arr]
        if hasattr(res,'masks'):
            masks=getattr(res,'masks')
            arr=np.asarray(masks)
            if arr.ndim==2:
                return [arr.astype(bool)]
            return [np.squeeze(np.asarray(m)).astype(bool) for m in arr]
        return []

def ensure(p:Path)->Path: p.mkdir(parents=True,exist_ok=True); return p
def rgba_to_rgb(x): return np.ascontiguousarray(x[...,:3].astype(np.uint8))
def depth_vis(d):
    ok=np.isfinite(d)&(d>0)
    if not np.any(ok): return np.zeros(d.shape,np.uint8)
    lo,hi=np.percentile(d[ok],[3,97]); x=np.clip((d-lo)/max(hi-lo,1e-6),0,1); x[~ok]=0; return (255*x).astype(np.uint8)
def largest_cc(mask):
    n,l,s,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    return mask if n<=1 else l==(int(np.argmax(s[1:,cv2.CC_STAT_AREA]))+1)
def score_mask(mask,depth,yaw):
    h,w=mask.shape; area=float(mask.sum()); area_ratio=area/(h*w)
    if area<=0: return ViewScore(yaw,0,0,0,0,0,0,0)
    n,labels,stats,centroids=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    if n<=1: largest,cx,cy=area,w/2,h/2; fragmentation_penalty=0.0
    else:
        comp_areas=stats[1:,cv2.CC_STAT_AREA].astype(np.float32)
        idx=int(np.argmax(comp_areas))+1; largest=float(stats[idx,cv2.CC_STAT_AREA]); cx,cy=centroids[idx]
        extra_area=float(comp_areas.sum()-largest)
        small_noise=float(comp_areas[comp_areas<0.08*largest].sum()) if largest>0 else 0.0
        fragmentation_penalty=np.clip(extra_area/max(area,1.0)+0.5*small_noise/max(area,1.0),0.0,1.0)
    conn=largest/max(area,1.0); area_s=float(math.exp(-((area_ratio-0.18)**2)/(2*0.12*0.12)))
    pos=float(math.exp(-((((cx/w)-0.5)/0.24)**2+(((cy/h)-0.56)/0.22)**2)/2.0))
    cnts,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    per=float(sum(cv2.arcLength(t,True) for t in cnts)); smooth=float(np.clip((4*math.pi*area)/max(per*per,1e-6),0,1))
    vd=depth[mask&np.isfinite(depth)&(depth>0)]; dep=0.0
    if vd.size>32:
        med=float(np.median(vd)); mad=float(np.median(np.abs(vd-med))); dep=float(np.clip(1.0-(mad/max(med,1e-6))/0.12,0,1))

    rows=np.nonzero(mask.any(axis=1))[0]
    cols=np.nonzero(mask.any(axis=0))[0]
    regularity=0.0; top_bias=0.0; leg_penalty=0.0
    if rows.size>0 and cols.size>0:
        widths=mask[rows].sum(axis=1).astype(np.float32)
        mean_width=float(np.mean(widths))
        width_std=float(np.std(widths))
        regularity=float(np.clip(1.0-width_std/max(mean_width,1.0),0.0,1.0))
        top_half=rows[:max(1,len(rows)//2)]
        bottom_half=rows[max(1,len(rows)//2):]
        top_mean=float(np.mean(mask[top_half].sum(axis=1))) if top_half.size>0 else 0.0
        bottom_mean=float(np.mean(mask[bottom_half].sum(axis=1))) if bottom_half.size>0 else 0.0
        top_bias=float(np.clip(top_mean/max(bottom_mean,1.0),0.0,2.0)/2.0)
        bottom_ratio=float(rows.max()/max(h-1,1))
        tail_ratio=float(bottom_mean/max(top_mean,1.0)) if top_mean>0 else 1.0
        leg_penalty=max(0.0,bottom_ratio-0.80)+max(0.0,tail_ratio-0.55)

    planar_mask=mask.copy().astype(np.uint8)
    if rows.size>4:
        kernel_w=max(5,int(round(mean_width*0.12))|1)
        kernel_h=max(5,int(round(len(rows)*0.18))|1)
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_w,kernel_h))
        planar_mask=cv2.morphologyEx(planar_mask,cv2.MORPH_OPEN,kernel)
        planar_mask=cv2.morphologyEx(planar_mask,cv2.MORPH_CLOSE,kernel)
        planar_mask=largest_cc(planar_mask.astype(bool)).astype(np.uint8)
    planar_ratio=float(planar_mask.sum()/max(area,1.0))
    hole_penalty=max(0.0,1.0-planar_ratio)

    fin=0.14*area_s+0.20*conn+0.12*pos+0.08*smooth+0.18*regularity+0.16*top_bias+0.20*dep+0.10*planar_ratio-0.16*leg_penalty-0.22*fragmentation_penalty-0.10*hole_penalty
    return ViewScore(yaw,area_s,conn,pos,smooth,dep,float(np.clip(fin,0.0,1.0)),float(area_ratio),False,regularity,top_bias,leg_penalty,fragmentation_penalty)

def overlay(rgb,mask,score):
    out=rgb.copy(); out[mask]=(0.55*out[mask]+0.45*np.array([30,220,80])).astype(np.uint8)
    cv2.putText(out,f'yaw={score.yaw_deg:.0f} score={score.final_score:.3f} area={score.area_ratio:.3f} depth={score.depth_score:.3f}',(20,36),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
    return out

def overlay_refined(rgb,raw_mask,connected_mask,score):
    out=rgb.copy()
    raw_only=raw_mask & ~connected_mask
    out[raw_only]=(0.65*out[raw_only]+0.35*np.array([255,180,20])).astype(np.uint8)
    out[connected_mask]=(0.50*out[connected_mask]+0.50*np.array([30,220,80])).astype(np.uint8)
    cv2.putText(out,f'refine yaw={score.yaw_deg:.1f} score={score.final_score:.3f} conn_px={int(connected_mask.sum())}',(20,36),cv2.FONT_HERSHEY_SIMPLEX,0.85,(255,255,255),2)
    return out

def dirs(pitch_deg,yaw_step_deg):
    p=math.radians(pitch_deg)
    for yaw in np.arange(0.0,360.0,yaw_step_deg):
        y=math.radians(float(yaw)); d=np.array([math.cos(p)*math.cos(y),math.cos(p)*math.sin(y),math.sin(p)],np.float32); d/=np.linalg.norm(d)+1e-8; yield float(yaw),d

def select_contiguous_views(scores, top_k, min_score):
    n=len(scores)
    if n==0:
        return []
    k=max(1,min(top_k,n))
    score_values=[s.final_score if s.final_score>=min_score else 0.0 for s in scores]
    doubled=score_values+score_values
    prefix=[0.0]
    for v in doubled:
        prefix.append(prefix[-1]+v)
    best_start=0; best_sum=-1.0
    for start in range(n):
        total=prefix[start+k]-prefix[start]
        if total>best_sum:
            best_sum=total; best_start=start
    return [int((best_start+i)%n) for i in range(k)]

def angular_distance_deg(a: float, b: float) -> float:
    return abs(((a-b+180.0)%360.0)-180.0)

def refine_yaws_from_kept(scores, keep, coarse_step, refine_step, margin):
    if not keep:
        return []
    kept_yaws=[float(scores[i].yaw_deg)%360.0 for i in keep]
    half_window=max(float(coarse_step)*0.5+float(margin), float(refine_step))
    candidates=[]
    for yaw in kept_yaws:
        offsets=np.arange(-half_window, half_window+0.5*refine_step, refine_step)
        for off in offsets:
            y=float((yaw+off)%360.0)
            if all(angular_distance_deg(y, old)>0.25*refine_step for old in candidates):
                candidates.append(y)
    candidates.sort(key=lambda y: angular_distance_deg(y, kept_yaws[0]))
    return candidates

def largest_depth_connected_mask(mask: np.ndarray, depth: np.ndarray, min_pixels: int = 64) -> np.ndarray:
    valid=mask & np.isfinite(depth) & (depth>0)
    if int(valid.sum())<min_pixels:
        return valid
    n,labels,stats,_=cv2.connectedComponentsWithStats(valid.astype(np.uint8),8)
    if n<=1:
        return valid
    best_label=0; best_area=-1
    for label in range(1,n):
        comp=labels==label
        vals=depth[comp]
        if vals.size<min_pixels:
            continue
        med=float(np.median(vals)); mad=float(np.median(np.abs(vals-med)))
        depth_consistency=max(0.05,1.0-mad/max(med,1e-6))
        area=int(stats[label,cv2.CC_STAT_AREA])
        score=area*depth_consistency
        if score>best_area:
            best_area=score; best_label=label
    if best_label<=0:
        return valid
    return labels==best_label

def planarize_mask(mask: np.ndarray) -> np.ndarray:
    rows=np.nonzero(mask.any(axis=1))[0]
    if rows.size<=4:
        return largest_cc(mask.astype(bool))
    widths=mask[rows].sum(axis=1).astype(np.float32)
    mean_width=float(np.mean(widths))
    kernel_w=max(5,int(round(mean_width*0.12))|1)
    kernel_h=max(5,int(round(len(rows)*0.18))|1)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_w,kernel_h))
    out=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_OPEN,kernel)
    out=cv2.morphologyEx(out,cv2.MORPH_CLOSE,kernel)
    return largest_cc(out.astype(bool))

def validate_mesh_depth(cam, sim, rgb_dir, depth_dir):
    set_scene_visibility(show_mesh=True, show_gauss=False)
    sim.step(render=True)
    cam.update(sim.get_physics_dt(),force_recompute=True)
    rgb=rgba_to_rgb(cam.data.output['rgb'][0].detach().cpu().numpy())
    depth=cam.data.output['distance_to_image_plane'][0].detach().cpu().numpy().squeeze()
    valid_depth=np.isfinite(depth)&(depth>0)
    cv2.imwrite(str(rgb_dir/'mesh_depth_rgb.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(depth_dir/'mesh_depth_only.png'),depth_vis(depth))
    stats={
        'valid_depth_pixels': int(valid_depth.sum()),
        'depth_min': None if not np.any(valid_depth) else float(depth[valid_depth].min()),
        'depth_max': None if not np.any(valid_depth) else float(depth[valid_depth].max()),
        'depth_mean': None if not np.any(valid_depth) else float(depth[valid_depth].mean()),
    }
    return stats


def capture_rgb_and_depth(cam, sim):
    set_scene_visibility(show_mesh=False, show_gauss=True)
    sim.step(render=True)
    cam.update(sim.get_physics_dt(),force_recompute=True)
    rgb=rgba_to_rgb(cam.data.output['rgb'][0].detach().cpu().numpy())

    set_scene_visibility(show_mesh=True, show_gauss=False)
    sim.step(render=True)
    cam.update(sim.get_physics_dt(),force_recompute=True)
    depth=cam.data.output['distance_to_image_plane'][0].detach().cpu().numpy().squeeze()

    return rgb, depth

def backproject(mask,depth,K,pos_w,quat_ros,max_points):
    ok=mask&np.isfinite(depth)&(depth>0); vs,us=np.nonzero(ok)
    if us.size==0: return np.empty((0,3),np.float32)
    if us.size>max_points:
        sel=np.linspace(0,us.size-1,max_points,dtype=np.int64); us,vs=us[sel],vs[sel]
    z=depth[vs,us].astype(np.float32); fx,fy,cx,cy=float(K[0,0]),float(K[1,1]),float(K[0,2]),float(K[1,2])
    x=((us.astype(np.float32)-cx)*z)/max(fx,1e-6); y=((vs.astype(np.float32)-cy)*z)/max(fy,1e-6)
    pts=np.stack([x,y,z],1); R=matrix_from_quat(torch.from_numpy(quat_ros.astype(np.float32)).unsqueeze(0))[0].cpu().numpy()
    return ((R@pts.T).T+pos_w.reshape(1,3)).astype(np.float32)
def pcd(points,color=None):
    c=o3d.geometry.PointCloud();
    if points.size==0: return c
    c.points=o3d.utility.Vector3dVector(points.astype(np.float64))
    if color is not None: c.colors=o3d.utility.Vector3dVector(np.tile(np.array(color,dtype=np.float64),(points.shape[0],1)))
    return c

def cluster_main(points,voxel,eps,min_points):
    if points.size==0: return np.empty((0,3),np.float32),[]
    c=pcd(points); c=c.voxel_down_sample(voxel) if voxel>0 else c; pts=np.asarray(c.points,dtype=np.float32)
    if pts.size==0: return np.empty((0,3),np.float32),[]
    labels=np.array(c.cluster_dbscan(eps=eps,min_points=min_points,print_progress=False),dtype=np.int32); stats=[]
    for label in [int(x) for x in np.unique(labels) if x>=0]:
        q=pts[labels==label]
        if q.shape[0]<3: continue
        xy_min,xy_max=q[:,:2].min(0),q[:,:2].max(0); xy_area=float(np.prod(np.maximum(xy_max-xy_min,1e-6))); z_std,z_med=float(np.std(q[:,2])),float(np.median(q[:,2]))
        cov=np.cov((q-q.mean(0)).T); eigvals,eigvecs=np.linalg.eigh(cov); up=float(abs(eigvecs[:,int(np.argmin(eigvals))][2]))
        score=0.45*min(q.shape[0]/3000.0,1.0)+0.20*min(xy_area/0.25,1.0)+0.20*up+0.15*max(0.0,1.0-z_std/0.04)
        stats.append(ClusterStat(label,int(q.shape[0]),z_med,z_std,xy_area,up,float(score)))
    stats.sort(key=lambda s:s.final_score,reverse=True)
    return (pts if not stats else pts[labels==stats[0].cluster_id]),stats

def shutdown_app_and_exit(code: int = 0):
    # Isaac/Kit teardown can hang after all outputs are written. For this batch
    # debug script we prefer an immediate process exit so shell pipelines can
    # continue to Phase 2 reliably.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)

def main():
    scene=Path(args.scene_usd)
    if not scene.exists(): raise FileNotFoundError(scene)
    out=ensure(Path(args.output_dir))
    rgb_dir=ensure(out/'rgb') if args.save_debug_images or args.validate_mesh_depth_only else None
    depth_dir=ensure(out/'depth') if args.save_debug_images or args.validate_mesh_depth_only else None
    mask_dir=ensure(out/'mask') if args.save_debug_images else None
    overlay_dir=ensure(out/'overlay') if args.save_debug_images else None
    pts_dir=ensure(out/'points') if args.save_debug_images else None
    det=LangSAMDetector(); sim=SimulationContext(sim_utils.SimulationCfg(dt=1/60)); sim.set_camera_view(eye=[2,2,2],target=[0,0,0])
    usd_cfg=sim_utils.UsdFileCfg(usd_path=str(scene)); usd_cfg.func('/World/Scene',usd_cfg,translation=(0,0,0),orientation=(1,0,0,0))
    cam=Camera(CameraCfg(prim_path='/World/debug_camera',update_period=0.0,height=args.image_height,width=args.image_width,data_types=['rgb','distance_to_image_plane'],update_latest_camera_pose=True,spawn=sim_utils.PinholeCameraCfg(focal_length=14.0,focus_distance=400.0,horizontal_aperture=24.0,clipping_range=(0.01,120.0)),offset=CameraCfg.OffsetCfg(pos=(0,0,0),rot=(1,0,0,0),convention='ros')))
    sim.reset(); apply_default_render_settings(task_name='real2sim_phase1'); set_scene_visibility(show_mesh=False, show_gauss=True); cam.reset(); sim.step(render=True); cam.update(sim.get_physics_dt(),force_recompute=True); origin=np.array(args.camera_origin,dtype=np.float32); scores=[]; masks=[]; depths=[]; poses=[]; rgbs=[]; merged=[]
    if args.validate_mesh_depth_only:
        eye=torch.tensor(origin,dtype=torch.float32,device=sim.device).unsqueeze(0)
        target=torch.tensor(origin+np.array([1.0,0.0,1.0],dtype=np.float32),dtype=torch.float32,device=sim.device).unsqueeze(0)
        cam.set_world_poses_from_view(eye,target)
        stats=validate_mesh_depth(cam,sim,rgb_dir,depth_dir)
        summary={'scene_usd':str(scene),'mesh_depth_validation':stats}
        (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
        print(f"[MeshDepth] valid_depth_pixels={stats['valid_depth_pixels']} depth_min={stats['depth_min']} depth_max={stats['depth_max']}")
        shutdown_app_and_exit()
    for i,(yaw,d) in enumerate(dirs(args.pitch_deg,args.yaw_step_deg)):
        eye=torch.tensor(origin,dtype=torch.float32,device=sim.device).unsqueeze(0); target=torch.tensor(origin+d,dtype=torch.float32,device=sim.device).unsqueeze(0)
        cam.set_world_poses_from_view(eye,target)
        rgb,depth=capture_rgb_and_depth(cam,sim)
        best_mask=np.zeros(depth.shape,bool); best_score=ViewScore(yaw,0,0,0,0,0,0,0)
        for m in det.predict(rgb,args.sam_prompt):
            m=planarize_mask(largest_cc(m.astype(bool))); s=score_mask(m,depth,yaw)
            if s.final_score>best_score.final_score: best_mask,best_score=m,s
        scores.append(best_score); masks.append(best_mask); depths.append(depth); rgbs.append(rgb); poses.append({'K':cam.data.intrinsic_matrices[0].detach().cpu().numpy(),'pos':cam.data.pos_w[0].detach().cpu().numpy(),'quat':cam.data.quat_w_ros[0].detach().cpu().numpy()})
        if args.depth_debug:
            valid_depth = np.isfinite(depth) & (depth > 0)
            overlap = best_mask & valid_depth
            print(f"[DepthDebug] view_{i:02d} yaw={yaw:.0f} mask={int(best_mask.sum())} valid_depth={int(valid_depth.sum())} overlap={int(overlap.sum())} score={best_score.final_score:.3f} regularity={best_score.regularity_score:.3f} top_bias={best_score.top_bias_score:.3f} leg_penalty={best_score.leg_penalty:.3f} fragmentation={best_score.fragmentation_penalty:.3f}")
        if args.save_debug_images:
            cv2.imwrite(str(rgb_dir/f'view_{i:02d}.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)); cv2.imwrite(str(depth_dir/f'view_{i:02d}.png'),depth_vis(depth)); cv2.imwrite(str(mask_dir/f'view_{i:02d}.png'),255*best_mask.astype(np.uint8)); cv2.imwrite(str(overlay_dir/f'view_{i:02d}.png'),cv2.cvtColor(overlay(rgb,best_mask,best_score),cv2.COLOR_RGB2BGR))
    keep=select_contiguous_views(scores,args.top_k_views,args.min_view_score)
    if not keep and scores: keep=[int(np.argmax([s.final_score for s in scores]))]
    for i in keep:
        scores[i].kept=True

    connected_records=[]; reference_record=None
    reference_dir=ensure(out/'reference') if args.save_debug_images else out
    for i in keep:
        conn=largest_depth_connected_mask(masks[i],depths[i])
        conn_score=score_mask(conn,depths[i],scores[i].yaw_deg)
        pts=backproject(conn,depths[i],poses[i]['K'],poses[i]['pos'],poses[i]['quat'],args.max_points_per_view)
        connected_records.append({'idx':i,'yaw':float(scores[i].yaw_deg),'score':conn_score,'raw_mask':masks[i],'conn':conn,'pts':pts,'rgb':rgbs[i]})
        if args.save_debug_images:
            cv2.imwrite(str(mask_dir/f'view_{i:02d}_largest_connected.png'),255*conn.astype(np.uint8))
            rgb_debug=cv2.imread(str(rgb_dir/f'view_{i:02d}.png'))
            if rgb_debug is not None:
                cv2.imwrite(str(overlay_dir/f'view_{i:02d}_largest_connected.png'),cv2.cvtColor(overlay_refined(cv2.cvtColor(rgb_debug,cv2.COLOR_BGR2RGB),masks[i],conn,conn_score),cv2.COLOR_RGB2BGR))
            o3d.io.write_point_cloud(str(pts_dir/f'view_{i:02d}_largest_connected.ply'),pcd(pts,(0.1,0.9,0.2)))
        print(f"[Connected] view_{i:02d} yaw={scores[i].yaw_deg:.1f} raw_mask={int(masks[i].sum())} conn={int(conn.sum())} score={conn_score.final_score:.3f} points={pts.shape[0]}")
    if connected_records:
        connected_records.sort(key=lambda r:(int(r['conn'].sum()), r['score'].final_score),reverse=True)
        reference_record=connected_records[0]
        merged.append(reference_record['pts'])
        ref_idx=reference_record['idx']
        ref_rgb=reference_record.get('rgb')
        if ref_rgb is not None:
            ref_overlay=overlay_refined(ref_rgb,reference_record['raw_mask'],reference_record['conn'],reference_record['score'])
            cv2.imwrite(str(reference_dir/'reference_overlay.png'),cv2.cvtColor(ref_overlay,cv2.COLOR_RGB2BGR))
            if args.save_debug_images:
                cv2.imwrite(str(reference_dir/'reference_rgb.png'),cv2.cvtColor(ref_rgb,cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(reference_dir/'reference_mask.png'),255*reference_record['conn'].astype(np.uint8))
        o3d.io.write_point_cloud(str(reference_dir/'reference_table_points.ply'),pcd(reference_record['pts'],(0.1,0.9,0.2)))
        print(f"[Reference] selected view_{ref_idx:02d} yaw={reference_record['yaw']:.1f} connected_pixels={int(reference_record['conn'].sum())}")
    if not merged:
        for i in keep:
            pts=backproject(masks[i],depths[i],poses[i]['K'],poses[i]['pos'],poses[i]['quat'],args.max_points_per_view); merged.append(pts); o3d.io.write_point_cloud(str(pts_dir/f'view_{i:02d}.ply'),pcd(pts,(0.2,0.6,1.0)))
    merged_pts=np.concatenate(merged,0) if merged else np.empty((0,3),np.float32); o3d.io.write_point_cloud(str(out/'table_candidates_merged.ply'),pcd(merged_pts,(0.7,0.7,0.7)))
    main_pts,cluster_stats=cluster_main(merged_pts,args.voxel_size,args.cluster_eps,args.cluster_min_points); o3d.io.write_point_cloud(str(out/'table_main_cluster.ply'),pcd(main_pts,(0.1,0.9,0.2)))
    reference_summary=None if reference_record is None else {'view_index':int(reference_record['idx']),'yaw_deg':float(reference_record['yaw']),'connected_pixels':int(reference_record['conn'].sum()),'score':asdict(reference_record['score']),'num_points':int(reference_record['pts'].shape[0]),'points':'reference/reference_table_points.ply' if args.save_debug_images else 'reference_table_points.ply'}
    summary={'scene_usd':str(scene),'camera_origin':list(args.camera_origin),'pitch_deg':args.pitch_deg,'yaw_step_deg':args.yaw_step_deg,'sam_prompt':args.sam_prompt,'kept_view_indices':keep,'reference_view':reference_summary,'connected_candidates':[{'view_index':int(r['idx']),'yaw_deg':float(r['yaw']),'connected_pixels':int(r['conn'].sum()),'score':asdict(r['score']),'num_points':int(r['pts'].shape[0])} for r in connected_records],'view_scores':[asdict(s) for s in scores],'num_merged_candidate_points':int(merged_pts.shape[0]),'num_main_cluster_points':int(main_pts.shape[0]),'table_points':'table_main_cluster.ply','reference_table_points':'reference/reference_table_points.ply' if args.save_debug_images else 'reference_table_points.ply','z_table_median':None if main_pts.size==0 else float(np.median(main_pts[:,2])),'xy_bounds':None if main_pts.size==0 else {'min':main_pts[:,:2].min(0).tolist(),'max':main_pts[:,:2].max(0).tolist()},'cluster_stats':[asdict(s) for s in cluster_stats]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print('[Phase1] tabletop detection complete')
    print(f'[Phase1] output_dir={out}')
    print(f'[Phase1] kept_views={keep}')
    print(f"[Phase1] z_table_median={summary['z_table_median']}")
    shutdown_app_and_exit()

if __name__=='__main__': main()
