#!/usr/bin/env python3
"""Generate self-contained SDF and calibration; Python standard library only."""
import argparse
import random
import json
import math
from pathlib import Path
import xml.etree.ElementTree as E
ROOT = Path(__file__).resolve().parents[1]
STYLES = {
    'dark': ('0.055 0.060 0.068 1', '0.025 0.028 0.035 1', 0.15, 0.7),
    'light': ('0.62 0.64 0.65 1', '0.36 0.38 0.40 1', 0.15, 0.7),
    'steel': ('0.38 0.40 0.43 1', '0.18 0.20 0.23 1', 0.7, 0.38),
    'sand': ('0.32 0.28 0.21 1', '0.14 0.13 0.11 1', 0.1, 0.8),
    'two-tone': ('0.06 0.065 0.075 1', '0.58 0.60 0.62 1', 0.2, 0.6),
}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--style', choices=STYLES, default='steel')
parser.add_argument('--seed', type=int, default=42, help='Fixed installation error seed')
parser.add_argument('--position-cm', type=float, default=3, help='Uniform +/- offset on each WORLD axis')
parser.add_argument('--roll-deg', type=float, default=0.5, help='World X rotation bound')
parser.add_argument('--pitch-deg', type=float, default=5, help='World Y rotation bound')
parser.add_argument('--yaw-deg', type=float, default=0.5, help='World Z rotation bound')
parser.add_argument('--ideal-cameras', action='store_true')
parser.add_argument('--tag-rover-inverted', action='store_true', help='Start tag rover upside down to test bottom ID 1')
parser.add_argument('--profile', choices=['demo_baseline','imx296_narrow','imx296_global_30','imx219_160'], default='imx296_global_30')
parser.add_argument('--layout', choices=['nadir','final','mounts5'], default='nadir',
                    help='nadir: every camera looks straight down; final: the tilt/roll layout '
                         'optimised for the IMX219-160 (corner cameras lean 10 deg toward the '
                         'arena centre line with the long image side along X, middle cameras '
                         'nadir with the long side along Y); mounts5: the five ceiling mounts '
                         '(3,3) (9,3) (3,9) (9,9) and a camera pair around (6,6), see '
                         'docs/camera_mounting.md')
parser.add_argument('--centre-spacing', type=float, default=0.30,
                    help='mounts5 only: distance between the two centre cameras, 0..1.0 m (fixture 1.1 x 1.1 m)')
parser.add_argument('--layout-variant', default='',
                    help='mounts5 only: comma-separated key=value, keys corner_tilt_deg (tilt of the '
                         'corner cameras toward (6,6)), corner_roll_deg (0: long side along X), '
                         'centre (xy: long sides perpendicular; xx: both along X; tilt_out_15: both '
                         'along X, leaning 15 deg away from each other)')
parser.add_argument('--lidar', choices=['none','airy'], default='none',
                    help='Add the ceiling RoboSense Airy at (6, 6, 2.75) looking down')
parser.add_argument('--world-name', default='mocap_arena',
                    help='Output worlds/<name>.sdf; configs go to config/ for the default name '
                         'and to config/<name>/ otherwise, so an existing world is never overwritten')
parser.add_argument('--lighting', choices=['colored', 'neutral'], default='colored')
parser.add_argument('--light-intensity', type=float, default=1.0, help='Scale of overhead lights, 0..4')
parser.add_argument('--shadow-lights', type=int, choices=range(10), default=2,
                    help='Number of spotlights casting shadows; all nine still illuminate the scene')
parser.add_argument('--output-dir', type=Path, default=ROOT,
                    help='Separate scenario directory containing worlds/ and config/')
args = parser.parse_args()
PROFILES={'demo_baseline':(1600,1200,15,'R8G8B8'),'imx296_narrow':(1440,1080,30,'R8G8B8'),'imx296_global_30':(1440,1080,30,'R8G8B8'),
          # Waveshare IMX219-160: full-field 2x2 binned mode, 8-bit, 83.7 fps max.
          'imx219_160':(1640,1232,80,'R8G8B8')}
# Fisheye profiles render through Gazebo's wideanglecamera with an ideal
# equidistant lens.  The datasheet gives 160 deg on the sensor diagonal, so the
# focal length is fixed by the diagonal and the image circle is cut at 80 deg.
FISHEYE={'imx219_160':dict(diag_fov_deg=160.0,cutoff_deg=80.0,env_texture=2048)}
image_width,image_height,camera_fps,pixel_format=PROFILES[args.profile]
fisheye=FISHEYE.get(args.profile)
for value in (args.position_cm, args.roll_deg, args.pitch_deg, args.yaw_deg):
    if not math.isfinite(value) or value < 0: parser.error('Error bounds must be finite and nonnegative')
if args.position_cm > 5: parser.error('position-cm must be <= 5 at this ceiling height')
if max(args.roll_deg, args.pitch_deg, args.yaw_deg) > 15: parser.error('Angle bounds must be <= 15 degrees')
if not math.isfinite(args.light_intensity) or not 0 <= args.light_intensity <= 4: parser.error('light-intensity must be 0..4')
MOUNTS5_VARIANT=dict(corner_tilt_deg=0.0,corner_roll_deg=0.0,centre='xy')
if args.layout!='mounts5' and (args.layout_variant or args.centre_spacing!=0.30):
    parser.error('--layout-variant and --centre-spacing apply to --layout mounts5 only')
for item in filter(None,args.layout_variant.split(',')):
    key,_,value=item.partition('=')
    if key not in MOUNTS5_VARIANT: parser.error(f'unknown layout variant key {key!r}')
    if key=='centre':
        if value not in ('xy','xx','tilt_out_15'): parser.error('centre must be xy, xx or tilt_out_15')
        MOUNTS5_VARIANT[key]=value
    else: MOUNTS5_VARIANT[key]=float(value)
if not 0<=args.centre_spacing<=1.0: parser.error('centre-spacing must be 0..1.0 m')
OUTPUT = args.output_dir.resolve()
CONFIG_DIR = OUTPUT / 'config' if args.world_name == 'mocap_arena' else OUTPUT / 'config' / args.world_name
for directory in (OUTPUT / 'worlds', CONFIG_DIR): directory.mkdir(parents=True, exist_ok=True)
rng = random.Random(args.seed)
nominal_cameras = []
def rotation(r, p, y):
    cr,sr,cp,sp,cy,sy=math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    return [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
            [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]]
def matmul(a,b):
    return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
def transpose(a): return [[a[j][i] for j in range(3)] for i in range(3)]
def euler_zyx(R):
    """Inverse of rotation(): (roll, pitch, yaw) with R = Rz(yaw) Ry(pitch) Rx(roll)."""
    pitch=math.atan2(-R[2][0], math.hypot(R[0][0],R[1][0]))
    return (math.atan2(R[2][1],R[2][2]), pitch, math.atan2(R[1][0],R[0][0]))
NADIR=[[1,0,0],[0,-1,0],[0,0,-1]]   # optical x=+X, y=-Y, z=-Z (looking down)
def layout_rotation(pan,tilt,roll):
    """R_world_optical for a camera tilted ``tilt`` away from nadir toward the
    world direction ``pan`` (angle from +X), after spinning the sensor by
    ``roll`` about its own axis (0: long image side along world X)."""
    cr,sr=math.cos(roll),math.sin(roll)
    spin=[[cr,-sr,0],[sr,cr,0],[0,0,1]]
    ax=(-math.sin(pan),math.cos(pan),0.0)
    ct,st=math.cos(tilt),math.sin(tilt)
    K=[[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]]
    KK=matmul(K,K)
    Rt=[[(1 if i==j else 0)+st*K[i][j]+(1-ct)*KK[i][j] for j in range(3)] for i in range(3)]
    return matmul(Rt,matmul(NADIR,spin))
def layout_angles(layout,x,y):
    """(pan, tilt, roll) in radians for the camera at (x, y)."""
    if layout=='nadir': return (0.0,0.0,0.0)
    # pan is the axis the tilt rotates about, so the optical axis of the
    # y=2 cameras swings toward +Y and that of the y=10 cameras toward -Y:
    # every corner camera leans 10 deg toward the arena centre line y=6.
    if y<4: return (math.radians(270),math.radians(10),0.0)
    if y>8: return (math.radians(90),math.radians(10),0.0)
    return (0.0,0.0,math.radians(90))                            # nadir, long side along Y
def lean_pan(direction_deg):
    """pan for layout_rotation that swings the optical axis toward the world
    direction ``direction_deg`` (angle from +X): pan 270 leans toward +Y."""
    return math.radians((direction_deg+180)%360)
def camera_mounts():
    """[(x, y, (pan, tilt, roll))] in camera_1..6 order."""
    if args.layout!='mounts5':
        return [(x,y,layout_angles(args.layout,x,y)) for y in (2,6,10) for x in (3,9)]
    v=MOUNTS5_VARIANT; mounts=[]
    for x,y in ((3,3),(9,3),(3,9),(9,9)):
        mounts.append((x,y,(lean_pan(math.degrees(math.atan2(6-y,6-x))),
                            math.radians(v['corner_tilt_deg']),math.radians(v['corner_roll_deg']))))
    half=args.centre_spacing/2
    if v['centre']=='xy': centre=[(0.0,0.0,0.0),(0.0,0.0,math.radians(90))]
    elif v['centre']=='xx': centre=[(0.0,0.0,0.0),(0.0,0.0,0.0)]
    else: centre=[(lean_pan(180),math.radians(15),0.0),(lean_pan(0),math.radians(15),0.0)]
    return mounts+[(6-half,6,centre[0]),(6+half,6,centre[1])]

def el(p, tag, text=None, **attrs):
    e=E.SubElement(p,tag,attrs)
    if text is not None: e.text=str(text)
    return e

def box(link,name,size,pose,color,collision=True,transparency=0,metal=0.0,roughness=0.75):
    for kind in (['collision','visual'] if collision else ['visual']):
        v=el(link,kind,name=name+'_'+kind); el(v,'pose',pose); el(el(el(v,'geometry'),'box'),'size',size)
        if kind=='visual':
            m=el(v,'material'); el(m,'ambient',color); el(m,'diffuse',color); el(v,'transparency',transparency)
            pbr=el(el(m,'pbr'),'metal'); el(pbr,'metalness',metal); el(pbr,'roughness',roughness)
    return v

def plugin(p,file,name): return el(p,'plugin',filename='gz-sim-'+file+'-system',name='gz::sim::systems::'+name)
def inertia(link,mass,i):
    n=el(link,'inertial'); el(n,'mass',mass); a=el(n,'inertia')
    for k in ('ixx','iyy','izz'): el(a,k,i)
    for k in ('ixy','ixz','iyz'): el(a,k,0)

sdf=E.Element('sdf',version='1.9'); w=el(sdf,'world',name='mocap_arena')
p=el(w,'physics',name='physics',type='ignored'); el(p,'max_step_size',0.001); el(p,'real_time_factor',1)
for f,n in [('physics','Physics'),('user-commands','UserCommands'),('scene-broadcaster','SceneBroadcaster')]: plugin(w,f,n)
el(plugin(w,'sensors','Sensors'),'render_engine','ogre2')
scene=el(w,'scene'); el(scene,'ambient','0.18 0.19 0.22 1'); el(scene,'background','0.018 0.022 0.03 1'); el(scene,'shadows','true')
fill=el(w,'light',name='diffuse_fill',type='directional'); el(fill,'pose','6 6 2.8 0 0 0')
el(fill,'diffuse','0.62 0.66 0.74 1' if args.lighting=='colored' else '0.7 0.7 0.7 1'); el(fill,'direction','0.15 0.1 -1'); el(fill,'cast_shadows','false'); el(fill,'intensity',args.light_intensity)
# Colored illumination comes from actual lights, not painted patches on the floor.
light_colors = ['0.18 0.40 1 1', '1 0.18 0.48 1', '0.55 0.78 1 1',
                '0.65 0.28 1 1', '0.8 0.85 1 1', '0.14 0.75 0.72 1',
                '0.22 0.38 1 1', '1 0.35 0.18 1', '0.52 0.65 1 1']
a=el(w,'model',name='arena'); el(a,'static','true'); link=el(a,'link',name='structure')
box(link,'floor','12.2 12.2 0.1','6 6 -0.05 0 0 0','0.028 0.030 0.035 1',metal=0.15,roughness=0.58)
# Dark floor panel seams and repeatable wear; independent of installation seed.
wear = random.Random(17)
for k in range(1,6):
    for axis in ('x','y'):
        box(link,f'seam_{axis}_{k}','0.008 12 0.0005' if axis=='x' else '12 0.008 0.0005',
            f'{k*2 if axis=="x" else 6} {6 if axis=="x" else k*2} 0.0003 0 0 0',
            '0.012 0.014 0.016 1',False)
for k in range(90):
    sx,sy=wear.uniform(.15,11.85),wear.uniform(.15,11.85)
    box(link,f'scuff_{k}',f'{wear.uniform(.08,.4)} {wear.uniform(.002,.009)} 0.0002',
        f'{sx} {sy} 0.0007 0 0 {wear.uniform(-math.pi,math.pi)}','0.065 0.068 0.074 1',False)
for name,size,pose in [('south','12.2 0.1 3','6 -0.05 1.5 0 0 0'),('north','12.2 0.1 3','6 12.05 1.5 0 0 0'),('west','0.1 12 3','-0.05 6 1.5 0 0 0'),('east','0.1 12 3','12.05 6 1.5 0 0 0')]:
    box(link,name,size,pose,'0.055 0.07 0.09 1',transparency=0.88)
for side in range(4):
    horizontal=side<2; fixed=0.08 if side%2==0 else 11.92
    box(link,f'barrier_{side}','12 0.16 0.4' if horizontal else '0.16 12 0.4',
        f'{6 if horizontal else fixed} {fixed if horizontal else 6} .2 0 0 0','0.025 0.028 0.035 1')
    for k in range(30):
        # Alternating muted cyan / pink tape along the top rim, as in reference.
        box(link,f'tape_{side}_{k}','0.2 0.17 0.025' if horizontal else '0.17 0.2 0.025',
            f'{.2+k*.4 if horizontal else fixed} {fixed if horizontal else .2+k*.4} .407 0 0 0',
            '0.08 0.4 0.38 1' if k%2 else '0.48 0.12 0.25 1',False)
box(link,'ceiling','12.2 12.2 0.04','6 6 3.02 0 0 0','0.1 0.12 0.15 1',transparency=1)
for idx,(x,y) in enumerate(( (x,y) for y in (1.5,6,10.5) for x in (1.5,6,10.5) )):
    color=light_colors[idx] if args.lighting=='colored' else '0.85 0.85 0.85 1'
    light=el(w,'light',name=f'overhead_{idx}',type='spot'); el(light,'pose',f'{x} {y} 2.82 0 0 0')
    el(light,'intensity',args.light_intensity); el(light,'diffuse',color); el(light,'specular',color); el(light,'direction','0 0 -1'); el(light,'cast_shadows','true' if idx < args.shadow_lights else 'false')
    attenuation=el(light,'attenuation')
    for key,value in [('range',12),('constant',0.3),('linear',0.035),('quadratic',0.012)]: el(attenuation,key,value)
    spot=el(light,'spot'); el(spot,'inner_angle',0.48); el(spot,'outer_angle',1.12); el(spot,'falloff',1.2)
    visual=box(link,f'fixture_{idx}','0.28 0.18 0.035',f'{x} {y} 2.97 0 0 0',color,False)
    el(visual.find('material'),'emissive',color)

if fisheye:
    # Equidistant: r = f * theta.  The diagonal half-angle lands on the image corner.
    fx=(math.hypot(image_width,image_height)/2)/math.radians(fisheye['diag_fov_deg']/2)
    hfov=image_width/fx           # what scale_to_hfov needs to reproduce this f
    distortion=[0.0]*4; distortion_model='fisheye'
else:
    hfov=2*math.atan(8.2/(2*2.9)); fx=image_width/(2*math.tan(hfov/2))
    distortion=[0.0]*5; distortion_model='pinhole'
cameras=[]
for idx,(x,y,(pan,tilt,spin)) in enumerate(camera_mounts(),1):
    name=f'camera_{idx}'
    offset=[rng.uniform(-args.position_cm,args.position_cm)/100 for _ in range(3)]
    angles=[math.radians(rng.uniform(-bound,bound)) for bound in (args.roll_deg,args.pitch_deg,args.yaw_deg)]
    if args.ideal_cameras: offset=[0.0]*3; angles=[0.0]*3
    position=[v+d for v,d in zip([x,y,2.9],offset)]
    R_nominal=layout_rotation(pan,tilt,spin)
    # The sensor keeps its (0, pi/2, pi/2) pose, which realises NADIR; the
    # layout goes onto the model pose so the installation error still
    # composes on the world side exactly as the convention below states.
    R_layout=matmul(R_nominal,transpose(NADIR))
    model_rpy=list(euler_zyx(matmul(rotation(*angles),R_layout)))
    m=el(w,'model',name=name); el(m,'static','true'); el(m,'pose',' '.join(map(str,position+model_rpy))); l=el(m,'link',name='camera_link')
    box(l,'housing','0.12 0.08 0.06','0 0 0.05 0 0 0','0.12 0.12 0.15 1',False)
    sensor=el(l,'sensor',name=name,type='wideanglecamera' if fisheye else 'camera'); el(sensor,'pose',f'0 0 0 0 {math.pi/2} {math.pi/2}'); el(sensor,'always_on','true'); el(sensor,'update_rate',camera_fps); el(sensor,'topic',f'/cameras/{name}/image'); el(sensor,'visualize','true')
    c=el(sensor,'camera'); el(c,'horizontal_fov',hfov); el(c,'camera_info_topic',f'/cameras/{name}/camera_info')
    im=el(c,'image'); el(im,'width',image_width); el(im,'height',image_height); el(im,'format',pixel_format); clip=el(c,'clip'); el(clip,'near',0.05); el(clip,'far',20)
    if fisheye:
        lens=el(c,'lens'); el(lens,'type','equidistant'); el(lens,'scale_to_hfov','true')
        el(lens,'cutoff_angle',math.radians(fisheye['cutoff_deg'])); el(lens,'env_texture_size',fisheye['env_texture'])
        intr=el(lens,'intrinsics'); el(intr,'fx',fx); el(intr,'fy',fx); el(intr,'cx',image_width/2); el(intr,'cy',image_height/2); el(intr,'s',0)
    cameras.append(dict(name=name,position_world=[x,y,2.9],R_world_optical=R_nominal,K=[fx,0,image_width/2,0,fx,image_height/2,0,0,1],D=distortion,distortion_model=distortion_model,image_size=[image_width,image_height],horizontal_fov=hfov,
                        layout=dict(pan_deg=math.degrees(pan),tilt_deg=math.degrees(tilt),roll_deg=math.degrees(spin)),
                        image_topic=f'/cameras/{name}/image',camera_info_topic=f'/cameras/{name}/camera_info'))
    nominal_cameras.append(cameras[-1].copy())
    cameras[-1]=dict(cameras[-1], position_world=position,
                     R_world_optical=matmul(rotation(*angles),R_nominal),
                     installation_offset_world_m=offset, installation_rpy_world_deg=list(map(math.degrees,angles)))
lidar=None
if args.lidar=='airy':
    # Same block as worlds/mocap_arena_l2.sdf after set_world_lidar --profile airy --rename.
    lidar=dict(name='arena_robosense_airy',position_world=[6,6,2.75],R_world_sensor=rotation(math.pi,0,0),
               topic='/robosense_airy/normal/scan',update_rate_hz=10.0,horizontal_samples=900,vertical_samples=95,
               range_min=0.1,range_max=60.0)
    lm=el(w,'model',name=lidar['name']); el(lm,'static','true'); el(lm,'pose','6 6 2.75 3.141592653589793 0 0')
    ll=el(lm,'link',name='airy_lidar_link')
    for part in ('base','rotor'):
        v=el(ll,'visual',name=f'l2_{part}_visual'); mesh=el(el(v,'geometry'),'mesh'); el(mesh,'scale','0.001 0.001 0.001')
        # Resolved through GZ_SIM_RESOURCE_PATH (which holds models/), so the
        # world runs from any clone location.
        el(mesh,'uri',f'model://unitree_l2/meshes/unitree_l2_{part}.obj')
    ls=el(ll,'sensor',name='l2_normal',type='gpu_lidar'); el(ls,'topic',lidar['topic']); el(ls,'always_on','true'); el(ls,'update_rate',lidar['update_rate_hz']); el(ls,'visualize','true')
    ray=el(ls,'ray'); scan=el(ray,'scan'); hz=el(scan,'horizontal'); el(hz,'samples',900); el(hz,'resolution',1); el(hz,'min_angle',-math.pi); el(hz,'max_angle',math.pi)
    vt=el(scan,'vertical'); el(vt,'samples',95); el(vt,'resolution',1); el(vt,'min_angle',0.0); el(vt,'max_angle',math.pi/2)
    rg=el(ray,'range'); el(rg,'min',0.1); el(rg,'max',60.0); el(rg,'resolution',0.015)

for name,x,y,color,tag in [('tag_rover',3,2,'0.075 0.09 0.11 1',True),('yolo_rover',9,6,STYLES[args.style][0],False)]:
    m=el(w,'model',name=name); el(m,'pose',f'{x} {y} 0.226 {math.pi} 0 0' if tag and args.tag_rover_inverted else f'{x} {y} 0.14 0 0 0'); l=el(m,'link',name='base_link'); inertia(l,8,0.22)
    box(l,'body','0.72 0.52 0.18' if tag else '0.9 0.52 0.18','0 0 0.12 0 0 0',color,metal=0.2 if tag else STYLES[args.style][2],roughness=0.7 if tag else STYLES[args.style][3])
    if tag:
        box(l,'tag_board','0.5 0.5 0.01','0 0 0.22 0 0 0','1 1 1 1')
        # Official AprilRobotics bitmaps; bottom frame is rotated pi about base X.
        patterns={
            0:['0000000000','0111111110','0100101010','0110001010','0110011110','0101011110','0110100110','0111101110','0111111110','0000000000'],
            1:['0000000000','0111111110','0100100110','0110100010','0100001110','0110011110','0101001010','0111011010','0111111110','0000000000'],
        }
        box(l,'bottom_tag_board','0.5 0.5 0.01','0 0 0.02 0 0 0','1 1 1 1')
        for tag_id,bits in patterns.items():
            # OpenCV's canonical AprilTag dictionary axes are 180 degrees
            # from these source bitmaps. Rotate the PRINT, preserving the
            # configured tag->base axes and the bottom face's proper rotation.
            bits=[line[::-1] for line in bits[::-1]]
            for row,line in enumerate(bits):
                for col,bit in enumerate(line):
                    if bit=='1':
                        cell_x=(col-4.5)*0.05
                        cell_y=(4.5-row)*0.05*(1 if tag_id==0 else -1)
                        cell_z=0.2252 if tag_id==0 else 0.0148
                        box(l,f'tag_{tag_id}_{row}_{col}','0.05 0.05 0.0004',
                            f'{cell_x} {cell_y} {cell_z} 0 0 0','0 0 0 1',False)
    else:
        box(l,'cabin','0.38 0.44 0.13','-0.12 0 0.275 0 0 0',STYLES[args.style][1],metal=STYLES[args.style][2],roughness=STYLES[args.style][3])
        box(l,'hood','0.26 0.44 0.04','0.3 0 0.23 0 0 0',color,metal=STYLES[args.style][2],roughness=STYLES[args.style][3])
        # Low armor and side rails give a less toy-like silhouette.
        for side in (-1,1):
            box(l,f'rail_{side}','0.87 0.035 0.055',f'0 {side*.25} .24 0 0 0',STYLES[args.style][1],False)
        box(l,'front_armor','0.22 0.53 0.025','0.4 0 .07 0 -0.35 0',STYLES[args.style][1],False)
        for k in range(5):
            box(l,f'vent_{k}','0.10 0.012 0.002',f'-.12 {(k-2)*.045} .3415 0 0 0','0.016 0.018 0.022 1',False)
    for side,wy in [('left',0.31),('right',-0.31)]:
        for end,wx in [('front',0.25),('rear',-0.25)]:
            wn=f'{side}_{end}'; wheel=el(m,'link',name=wn); el(wheel,'pose',f'{wx} {wy} 0 {-math.pi/2} 0 0'); inertia(wheel,0.5,0.004)
            for kind in ('collision','visual'):
                v=el(wheel,kind,name=wn+'_'+kind); g=el(el(v,'geometry'),'cylinder'); el(g,'radius',0.14); el(g,'length',0.09)
                if kind=='visual':
                    mat=el(v,'material'); el(mat,'ambient','0.035 0.035 0.035 1'); el(mat,'diffuse','0.035 0.035 0.035 1')
            joint=el(m,'joint',name=wn+'_joint',type='revolute'); el(joint,'parent','base_link'); el(joint,'child',wn); axis=el(joint,'axis'); el(axis,'xyz','0 0 1')
    d=plugin(m,'diff-drive','DiffDrive')
    for side in ('left','right'):
        for end in ('front','rear'): el(d,side+'_joint',f'{side}_{end}_joint')
    for key,value in [('wheel_separation',0.62),('wheel_radius',0.14),('topic',f'/model/{name}/cmd_vel'),('odom_topic',f'/model/{name}/odometry'),('max_linear_velocity',1),('min_linear_velocity',-1),('max_angular_velocity',2),('min_angular_velocity',-2)]: el(d,key,value)
    p=plugin(m,'pose-publisher','PosePublisher')
    for key,value in [('publish_link_pose','false'),('publish_sensor_pose','false'),('publish_collision_pose','false'),('publish_visual_pose','false'),('publish_model_pose','true'),('use_pose_vector_msg','true'),('update_frequency','30')]: el(p,key,value)
E.indent(sdf); WORLD_PATH=OUTPUT/'worlds'/f'{args.world_name}.sdf'; E.ElementTree(sdf).write(WORLD_PATH,encoding='utf-8',xml_declaration=True)
tags=[dict(family='tag36h11',id=0,size=0.4,placement='top',
           T_base_tag_translation=[0,0,0.2254],R_base_tag=[[1,0,0],[0,1,0],[0,0,1]]),
      dict(family='tag36h11',id=1,size=0.4,placement='bottom',
           T_base_tag_translation=[0,0,0.0146],R_base_tag=[[1,0,0],[0,-1,0],[0,0,-1]])]
# Keep legacy top-only entry; new consumers should use the tags list keyed by ID.
common=dict(world='mocap_arena',tag=tags[0],tags=tags,
            tag_frame_convention='x=image right, y=image up, z=outward normal; transforms map tag to base')
for filename,items,role in [('cameras.json',nominal_cameras,'nominal_uncalibrated'),('cameras_ground_truth.json',cameras,'ground_truth_evaluation_only')]:
    (CONFIG_DIR/filename).write_text(json.dumps(dict(common,cameras=items,role=role,lidar=lidar),indent=2)+'\n')
if args.ideal_cameras:
    # With no installation error the nominal geometry is exact, so it can be
    # handed to the runtime as a calibration; a perturbed world still needs
    # image calibration and gets no such file.
    (CONFIG_DIR/'runtime_cameras.json').write_text(json.dumps(dict(common,cameras=nominal_cameras,lidar=lidar,
        role='image_calibrated',calibration_version=f'{args.world_name}-ideal'),indent=2)+'\n')
# Sample coverage including perturbed extrinsics. No silent promise of full coverage.
coverage=[]
for height in (0,0.3654,0.5):
    uncovered=[]
    for ix in range(121):
        for iy in range(121):
            point=[ix/10,iy/10,height]; seen=False
            for cam in cameras:
                delta=[point[k]-cam['position_world'][k] for k in range(3)]
                r=cam['R_world_optical']; optical=[sum(r[k][j]*delta[k] for k in range(3)) for j in range(3)]
                if optical[2]<=0: continue
                if fisheye:
                    norm=math.sqrt(sum(v*v for v in optical)); theta=math.acos(optical[2]/norm)
                    rxy=math.hypot(optical[0],optical[1]); scale=fx*theta/rxy if rxy>1e-9 else 0.0
                    u=image_width/2+optical[0]*scale; v=image_height/2+optical[1]*scale
                    inside=theta<math.radians(fisheye['cutoff_deg']) and 0<=u<image_width and 0<=v<image_height
                else:
                    inside=abs(optical[0]/optical[2])<=math.tan(hfov/2) and abs(optical[1]/optical[2])<=.75*math.tan(hfov/2)
                if inside: seen=True; break
            if not seen: uncovered.append(point[:2])
    coverage.append(dict(height_m=height,sampled_points=14641,uncovered_count=len(uncovered),uncovered_examples=uncovered[:10]))
def visible_count(point,incidence_deg=65.0,margin_px=8.0):
    """Cameras that see ``point`` inside the image by ``margin_px`` and at an
    angle from the vertical (a flat marker's normal) below ``incidence_deg``."""
    count=0
    for cam in cameras:
        delta=[point[k]-cam['position_world'][k] for k in range(3)]
        norm=math.sqrt(sum(v*v for v in delta))
        if -delta[2]/norm<math.cos(math.radians(incidence_deg)): continue
        r=cam['R_world_optical']; optical=[sum(r[k][j]*delta[k] for k in range(3)) for j in range(3)]
        if optical[2]<=0: continue
        if fisheye:
            theta=math.acos(optical[2]/norm)
            if theta>=math.radians(fisheye['cutoff_deg']): continue
            rxy=math.hypot(optical[0],optical[1]); scale=fx*theta/rxy if rxy>1e-9 else 0.0
        else: scale=fx/optical[2]
        u=image_width/2+optical[0]*scale; v=image_height/2+optical[1]*scale
        if margin_px<=u<image_width-margin_px and margin_px<=v<image_height-margin_px: count+=1
    return count
coverage_fractions=[]
for height in (0.3654,0.483):
    counts=[visible_count([ix/10,iy/10,height]) for ix in range(121) for iy in range(121)]
    coverage_fractions.append(dict(height_m=height,incidence_gate_deg=65.0,edge_margin_px=8.0,
        **{f'at_least_{n}':sum(c>=n for c in counts)/len(counts) for n in (1,2,3)}))
settings=dict(profile=args.profile,layout=args.layout,lidar=args.lidar,world=str(WORLD_PATH),image_size=[image_width,image_height],camera_fps=camera_fps,lens=('equidistant' if fisheye else 'pinhole'),focal_px=fx,tag_rover_inverted=args.tag_rover_inverted,style=args.style,lighting=args.lighting,light_intensity=args.light_intensity,seed=args.seed,ideal_cameras=args.ideal_cameras,
              position_bound_cm=args.position_cm,rpy_bounds_deg=[args.roll_deg,args.pitch_deg,args.yaw_deg],
              rotation_convention='R_world_actual_optical = Rz(yaw) Ry(pitch) Rx(roll) R_world_nominal_optical',
              coverage_sample_step_m=0.1,geometric_coverage=coverage,coverage_fractions=coverage_fractions,
              **(dict(centre_spacing_m=args.centre_spacing,layout_variant=MOUNTS5_VARIANT) if args.layout=='mounts5' else {}))
(CONFIG_DIR/'scenario.json').write_text(json.dumps(settings,indent=2)+'\n')
print(f'Generated {WORLD_PATH}: {args.style}, {args.lighting}, seed={args.seed}, profile={args.profile}, layout={args.layout}, lidar={args.lidar}')
for item in coverage: print(f"  Z={item['height_m']}: {item['uncovered_count']} uncovered samples (ignores occlusion)")
for item in coverage_fractions:
    print(f"  Z={item['height_m']}: seen by >=1/>=2/>=3 cameras "
          f"{100*item['at_least_1']:.1f} / {100*item['at_least_2']:.1f} / {100*item['at_least_3']:.1f} % "
          f"(incidence <= 65 deg, 8 px from the edge)")
