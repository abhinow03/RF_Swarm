/* utils.js — small helpers used by several components.
   Anything used by only ONE component lives in that component's own
   file instead (see rf-emitter.js, scan-field.js). */
const COMPASS = ['N','NE','E','SE','S','SW','W','NW'];
const bearing = (vx,vz) => { let d = Math.atan2(vx,-vz)*180/Math.PI; return d<0 ? d+360 : d; };
const cardinal = d => COMPASS[Math.round(d/45)%8];
const clamp = (v,a,b) => Math.min(b, Math.max(a, v));

function makeCanvasPanel(w, h, pxPerMeter) {
  pxPerMeter = pxPerMeter || 2200;
  const cv = document.createElement('canvas');
  cv.width  = Math.round(w * pxPerMeter);
  cv.height = Math.round(h * pxPerMeter);
  const ctx = cv.getContext('2d');
  const tex = new THREE.CanvasTexture(cv);
  tex.anisotropy = 16;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(w, h),
    new THREE.MeshBasicMaterial({ map:tex, transparent:true, depthWrite:false, depthTest:false,
                                  fog:false, toneMapped:false })
  );
  mesh.renderOrder = 999;
  return { cv:cv, ctx:ctx, tex:tex, mesh:mesh, w:w, h:h };
}

function roundRect(ctx,x,y,w,h,r){
  ctx.beginPath();
  ctx.moveTo(x+r,y); ctx.arcTo(x+w,y,x+w,y+h,r); ctx.arcTo(x+w,y+h,x,y+h,r);
  ctx.arcTo(x,y+h,x,y,r); ctx.arcTo(x,y,x+w,y,r); ctx.closePath();
}
