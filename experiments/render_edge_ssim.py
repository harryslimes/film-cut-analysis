"""SSIM computed on EDGE maps (not intensity) of CLAHE-normalised frames.
Edges = where structure is; a colour flash changes edge *contrast* but not edge *location*,
so edge-SSIM should ignore the flash while still collapsing on a real cut. Renders a
decision sheet AND prints edge-SSIM vs intensity-SSIM so we can compare on the hard cases.
"""
import json, subprocess, argparse
import numpy as np, cv2

def ssim(a, b):
    a=a.astype(np.float32); b=b.astype(np.float32)
    ma=cv2.GaussianBlur(a,(7,7),1.5); mb=cv2.GaussianBlur(b,(7,7),1.5)
    va=cv2.GaussianBlur(a*a,(7,7),1.5)-ma**2; vb=cv2.GaussianBlur(b*b,(7,7),1.5)-mb**2
    vab=cv2.GaussianBlur(a*b,(7,7),1.5)-ma*mb; c1,c2=(2.55)**2,(7.65)**2
    return float((((2*ma*mb+c1)*(2*vab+c2))/((ma**2+mb**2+c1)*(va+vb+c2))).mean())

def edges(f):
    e=cv2.magnitude(cv2.Sobel(f,cv2.CV_32F,1,0,3),cv2.Sobel(f,cv2.CV_32F,0,1,3))
    return cv2.GaussianBlur(e,(3,3),0)                 # slight blur so SSIM is stable

ap=argparse.ArgumentParser()
ap.add_argument("--video",required=True); ap.add_argument("--cuts",required=True)
ap.add_argument("--start",type=float,required=True); ap.add_argument("--end",type=float,required=True)
ap.add_argument("--fps",type=float,default=23.976); ap.add_argument("--delta",type=float,default=0.2)
ap.add_argument("--thresh",type=float,default=0.5); ap.add_argument("--out",required=True)
ap.add_argument("--tw",type=int,default=160); ap.add_argument("--cols",type=int,default=4)
ap.add_argument("--win",type=int,default=4,help="frames each side for temporal median (0=single frame)")
a=ap.parse_args(); a0=a.start-1.0
def dec(sc): return subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-ss",str(a0),"-t",
    str(a.end-a0+1),"-i",a.video,"-vf",sc,"-f","rawvideo","-"],stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout
gray=np.frombuffer(dec("scale=256:144,format=gray"),np.uint8).reshape([-1,144,256])
pr=subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-show_entries",
    "stream=width,height","-of","csv=p=0",a.video]).decode().strip().split(","); W,H=int(pr[0]),int(pr[1]); th=int(round(a.tw*H/W/2)*2)
col=np.frombuffer(dec(f"scale={a.tw}:{th},format=bgr24"),np.uint8).reshape([-1,th,a.tw,3])
clahe=cv2.createCLAHE(2.0,(8,8)); grayc=np.stack([clahe.apply(f) for f in gray]).astype(np.float32)
cuts=[c for c in json.load(open(a.cuts))["cuts"] if a.start<=c<=a.end]; d=int(round(a.delta*a.fps)); W=a.win
n=len(grayc)
def med(i0,i1):  # temporal median of CLAHE frames in [i0,i1) -> kills transient flashes
    i0=max(0,i0); i1=min(n,max(i1,i0+1)); return np.median(grayc[i0:i1],axis=0)
pad,lab=8,20; pair_w=a.tw*2+pad; cw=pair_w+pad*2; ch=th+lab+pad*2
cols=a.cols; rows=(len(cuts)+cols-1)//cols; sheet=np.full((rows*ch+pad,cols*cw+pad,3),30,np.uint8); nk=0
print(f"{'#':>3} {'time':>7} {'edgeSSIM':>8} {'intSSIM':>7}  (win={W} median)")
for k,c in enumerate(cuts):
    ig=int(round((c-a0)*a.fps)); ib,ia=max(0,ig-d),min(n-1,ig+d)
    if W>0:  # temporal median each side, gap of d around the cut
        gb=med(ig-d-W, ig-d); ga=med(ig+d, ig+d+W)
    else:
        gb=grayc[ib]; ga=grayc[ia]
    es=ssim(edges(gb),edges(ga)); is_=ssim(gb,ga)
    keep= es < a.thresh; nk+=keep
    print(f"{k+1:3d} {c:7.1f} {es:8.3f} {is_:7.3f}")
    r,cc=divmod(k,cols); y=pad+r*ch+lab; x=pad+cc*cw
    sheet[y:y+th,x:x+a.tw]=col[ib]; sheet[y:y+th,x+a.tw+pad:x+2*a.tw+pad]=col[ia]
    color=(0,220,0) if keep else (0,0,230)
    cv2.rectangle(sheet,(x-2,y-2),(x+pair_w+2,y+th+2),color,2)
    cv2.putText(sheet,f"#{k+1} {c:.1f} e{es:.2f} {'KEEP' if keep else 'drop'}",(x,pad+r*ch+14),
                cv2.FONT_HERSHEY_SIMPLEX,0.4,color,1,cv2.LINE_AA)
cv2.imwrite(a.out,sheet)
print(f"\n{len(cuts)} cuts -> keep {nk}, drop {len(cuts)-nk} (edgeSSIM<{a.thresh}) -> {a.out}")
