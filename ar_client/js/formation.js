/* formation.js — formation geometry (ported 1:1 from capstone.py),
   the cosine S-curve blend, and the geometric shape classifier. */
/* ============================================================
   Formation geometry — ported 1:1 from the capstone's Python
   get_formation_offsets(). Same numbers, same units (metres),
   same coordinate convention (x=right, y=forward -> mapped here
   to scene x=east, z=-north; z=up -> scene y=up).
   dispersed/converging are randomized scatter by design (matches
   the Python: a fresh np.random.default_rng() draw per call).
   ============================================================ */

const FORMATION_NAMES = ['v_shape','encirclement','column','diamond','dispersed','converging','shield'];

function randomBoxScatter() {
  // low=[-20,-20,-10], high=[20,20,10] on [right,forward,up] -> [x,z,y] here
  const pts = [];
  for (let i=0;i<6;i++) {
    pts.push([
      -20 + Math.random()*40,   // x (right)
      -10 + Math.random()*20,   // y (up)
      -20 + Math.random()*40    // z (forward, sign handled by caller)
    ]);
  }
  return pts;
}

function getFormationOffsets(type, spread) {
  spread = spread === undefined ? 1.0 : spread;
  let off;   // array of 6x [x,y,z] in SCENE space (x=right, y=up, z=-forward)
  if (type === 'v_shape') {
    off = [[0,0,0],[-5,0,5],[5,0,5],[-10,0,10],[10,0,10],[0,0,8]];
  } else if (type === 'encirclement') {
    off = [];
    for (let i=0;i<6;i++){ const a = (2*Math.PI*i)/6; off.push([10*Math.cos(a), 0, -10*Math.sin(a)]); }
  } else if (type === 'column') {
    off = [[0,0,0],[0,0,5],[0,0,10],[0,0,15],[0,0,20],[0,0,25]];
  } else if (type === 'diamond') {
    off = [[0,10,0],[0,0,-10],[-10,0,0],[10,0,0],[0,0,10],[0,-10,0]];
  } else if (type === 'dispersed' || type === 'converging') {
    off = randomBoxScatter().map(p=>[p[0], p[1], -p[2]]);
  } else if (type === 'shield') {
    off = [[-10,5,-10],[0,5,-10],[10,5,-10],[-10,-5,-10],[0,-5,-10],[10,-5,-10]];
  } else {
    throw new Error('unknown formation: ' + type);
  }
  return off.map(p => [p[0]*spread, p[1]*spread, p[2]*spread]);
}

/* exact cosine S-curve blend from the transition generator:
   alpha=0 -> pure A, alpha=1 -> pure B, physically-plausible ease. */
function cosineRamp(progress) {
  progress = clamp(progress, 0, 1);
  return (1 - Math.cos(Math.PI * progress)) / 2;
}
function blendOffsets(a, b, alpha) {
  return a.map((p,i) => [
    p[0]*(1-alpha) + b[i][0]*alpha,
    p[1]*(1-alpha) + b[i][1]*alpha,
    p[2]*(1-alpha) + b[i][2]*alpha
  ]);
}

function shapeDescriptor(points) {
  // points: 6x [x,y,z] in a shared frame. Horizontal-plane (x,z) shape,
  // recentered and scale-normalized so absolute position/size don't matter.
  let cx=0, cz=0;
  points.forEach(p=>{ cx+=p[0]; cz+=p[2]; });
  cx/=points.length; cz/=points.length;
  const pts = points.map(p=>[p[0]-cx, p[2]-cz]);

  const radii = pts.map(p=>Math.hypot(p[0],p[1])).sort((a,b)=>a-b);
  const meanR = (radii.reduce((a,b)=>a+b,0)/radii.length) || 1;
  const radiiN = radii.map(r=>r/meanR);

  const nn = pts.map((p,i)=>{
    let m=Infinity;
    pts.forEach((q,j)=>{ if(i!==j){ const d=Math.hypot(p[0]-q[0],p[1]-q[1]); if(d<m)m=d; } });
    return m/meanR;
  }).sort((a,b)=>a-b);

  let sxx=0,szz=0,sxz=0;
  pts.forEach(p=>{ sxx+=p[0]*p[0]; szz+=p[1]*p[1]; sxz+=p[0]*p[1]; });
  sxx/=pts.length; szz/=pts.length; sxz/=pts.length;
  const tr=sxx+szz, det=sxx*szz-sxz*sxz;
  const disc=Math.sqrt(Math.max(0, tr*tr/4-det));
  const l1=tr/2+disc, l2=Math.max(1e-6, tr/2-disc);
  const aspect = Math.sqrt(l1/l2);

  const hull = convexHull2D(pts);
  const hullArea = Math.abs(polygonArea(hull)) / (meanR*meanR || 1);

  return radiiN.concat(nn).concat([aspect, hullArea]);
}

function convexHull2D(pts) {
  const p = pts.slice().sort((a,b)=> a[0]===b[0] ? a[1]-b[1] : a[0]-b[0]);
  const cross=(o,a,b)=>(a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0]);
  const lower=[]; for (const pt of p){ while(lower.length>=2 && cross(lower[lower.length-2],lower[lower.length-1],pt)<=0) lower.pop(); lower.push(pt); }
  const upper=[]; for (let i=p.length-1;i>=0;i--){ const pt=p[i]; while(upper.length>=2 && cross(upper[upper.length-2],upper[upper.length-1],pt)<=0) upper.pop(); upper.push(pt); }
  lower.pop(); upper.pop();
  return lower.concat(upper);
}
function polygonArea(poly) {
  let a=0;
  for (let i=0;i<poly.length;i++){ const [x1,y1]=poly[i], [x2,y2]=poly[(i+1)%poly.length]; a += x1*y2 - x2*y1; }
  return a/2;
}

/* z-score normalization across the 7 templates, computed once, reused for
   every live frame so templates and live vectors are directly comparable. */
function buildClassifier(templatesByName) {
  const names = FORMATION_NAMES.filter(n => templatesByName[n]);
  const raw = names.map(n => shapeDescriptor(templatesByName[n]));
  const dims = raw[0].length;
  const mean = new Array(dims).fill(0), std = new Array(dims).fill(0);
  raw.forEach(v => v.forEach((x,i)=> mean[i]+=x));
  mean.forEach((m,i)=> mean[i]=m/raw.length);
  raw.forEach(v => v.forEach((x,i)=> std[i]+=(x-mean[i])*(x-mean[i])));
  std.forEach((s,i)=> std[i]=Math.sqrt(s/raw.length) || 1);
  const norm = v => v.map((x,i)=> (x-mean[i])/std[i]);
  const templatesN = raw.map(norm);

  return {
    names: names,
    classify: function (points) {
      const v = norm(shapeDescriptor(points));
      const dists = templatesN.map(t => Math.hypot(...t.map((x,i)=>x-v[i])));
      const w = dists.map(d => Math.exp(-d));
      const sum = w.reduce((a,b)=>a+b,0) || 1;
      const conf = w.map(x=>x/sum);
      let top=0; for (let i=1;i<conf.length;i++) if (conf[i]>conf[top]) top=i;
      return { label: names[top], confidences: names.map((n,i)=>[n, conf[i]]) };
    }
  };
}
