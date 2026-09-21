"""Deterministic S15 episode manifest generator (labels/evaluation only)."""
import math
STYLES=('steel','colored','dark','bright','striped')
def episode(name,seed,seconds=600,style='steel',lighting='neutral',brightness=1.0,occlusion=False,crossing=False):
    if style not in STYLES: raise ValueError('unknown rover style')
    points=[]; n=int(seconds*15)
    for i in range(n):
        t=i/15
        if name=='circle': x,y=6+2*math.cos(t*.2+seed),6+2*math.sin(t*.2+seed)
        elif name=='eight': x,y=6+2*math.sin(t*.2+seed),6+1.2*math.sin(2*(t*.2+seed))
        elif name=='straight': x,y=2+min(8,t*.2),2
        else: x,y=3,3
        points.append({'t_s':t,'position_m':[round(x,6),round(y,6)],'label_role':'evaluation_only'})
    return {'name':name,'seed':seed,'seconds':seconds,'style':style,'lighting':lighting,'brightness':brightness,'occlusion':occlusion,'crossing':crossing,'points':points}
def manifest(seeds=(42,7),seconds=600):
    rows=[]
    for seed in seeds:
        for name in ('rest','straight','circle','eight'):
            rows.append(episode(name,seed,seconds,STYLES[seed%len(STYLES)],'colored' if seed%2 else 'neutral',.5+(seed%3)*.25,name=='circle',name=='eight'))
    return {'schema':'sim-episodes-1','seed_set':list(seeds),'episodes':rows,'truth_role':'labels_and_evaluation_only'}
