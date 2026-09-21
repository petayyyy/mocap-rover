"""Reproducible image-derived extrinsic calibration experiment (evaluator truth hidden)."""
import math
import numpy as np

from localization_contracts.extrinsics import CalibrationGraph

def _T(x,y,z):
    t=np.eye(4); t[:3,3]=[x,y,z]; return t

def run(seed=42, noise_px=0.15):
    import cv2
    rng=np.random.default_rng(seed); K=np.array([[565.85,0,800],[0,565.85,600],[0,0,1.]],float); D=np.zeros((5,1))
    # Hidden installation pose: only the generator/evaluator knows this.
    truth=_T(3.08,1.94,2.86); truth[:3,:3]=np.diag([1.,-1.,-1.])
    graph=CalibrationGraph(['camera_1'])
    object_points=[]; holdout=[]
    for board_i, board_pose in enumerate((_T(0,0,0),_T(2,1,0),_T(7,4,0),_T(4,8,0))):
        points=np.array([[x,y,0] for x in np.linspace(-.8,.8,5) for y in np.linspace(-.6,.6,4)],float)
        Tcb=np.linalg.inv(truth) @ board_pose; rvec,_=cv2.Rodrigues(Tcb[:3,:3]); img,_=cv2.projectPoints(points,rvec,Tcb[:3,3],K,D)
        noisy=img.reshape(-1,2)+rng.normal(0,noise_px,img.reshape(-1,2).shape)
        graph.add_image_observation('camera_1',f'board_{board_i}',points,noisy,K,D,board_pose)
        object_points.append((points,board_pose))
    solved=graph.solve_image_observations(reprojection_limit_px=3.)['camera_1']
    estimated=np.asarray(solved.position_m); truth_position=truth[:3,3]
    # Independent held-out board: it was never registered with the solver.
    hold_points=np.array([[x,y,0] for x in np.linspace(-.7,.7,5) for y in np.linspace(-.5,.5,4)],float); hold_board=_T(1,7,0)
    T_est=np.eye(4); T_est[:3,:3]=np.asarray(solved.rotation).reshape(3,3); T_est[:3,3]=estimated
    Tcb_truth=np.linalg.inv(truth)@hold_board; rv_truth,_=cv2.Rodrigues(Tcb_truth[:3,:3]); hold_img,_=cv2.projectPoints(hold_points,rv_truth,Tcb_truth[:3,3],K,D)
    Tcb_est=np.linalg.inv(T_est)@hold_board; rv_est,_=cv2.Rodrigues(Tcb_est[:3,:3]); hold_pred,_=cv2.projectPoints(hold_points,rv_est,Tcb_est[:3,3],K,D)
    hold_error=float(np.sqrt(np.mean(np.sum((hold_img.reshape(-1,2)-hold_pred.reshape(-1,2))**2,axis=1))))
    return {'seed':seed,'independent_points':80,'held_out_points':20,'translation_error_m':float(np.linalg.norm(estimated-truth_position)),'held_out_reprojection_px':hold_error,
            'reprojection_quality':solved.quality,'noise_px':noise_px,'truth_used_only_by_generator_evaluator':True,'solver_reads_truth':False}
