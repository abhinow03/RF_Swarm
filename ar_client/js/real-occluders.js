/* real-occluders.js — turns the Scene Model (Space Setup) into invisible depth-writing geometry so real furniture occludes virtual content. */
AFRAME.registerComponent('real-occluders', {
  init: function () {
    this.entries = new Map();
    this.group = new THREE.Group();
    this.el.object3D.add(this.group);
    this.occMat = new THREE.MeshBasicMaterial({ colorWrite:false });
    this.occMat.depthWrite = true;
  },
  _planeGeometry: function (plane) {
    const pts = plane.polygon;
    if (!pts || pts.length < 3) return null;
    const verts = [], idx = [];
    for (let i=0;i<pts.length;i++) verts.push(pts[i].x, pts[i].y, pts[i].z);
    for (let i=1;i<pts.length-1;i++) idx.push(0, i, i+1);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
    g.setIndex(idx);
    return g;
  },
  _meshGeometry: function (xrMesh) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(xrMesh.vertices, 3));
    g.setIndex(new THREE.BufferAttribute(xrMesh.indices, 1));
    return g;
  },
  tick: function () {
    const S = window.RFX;
    const scene = this.el.sceneEl, frame = scene.frame, r = scene.renderer;
    this.group.visible = !!(S && S.isAR);
    if (!this.group.visible || !frame || !r || !r.xr) return;
    const ref = r.xr.getReferenceSpace();
    if (!ref) return;
    const seen = new Set();
    const self = this;
    function handle(item, space, buildGeom, changedTime) {
      seen.add(item);
      let e = self.entries.get(item);
      if (!e || (changedTime !== undefined && e.changed !== changedTime)) {
        const g = buildGeom(item);
        if (!g) return;
        if (e) { e.mesh.geometry.dispose(); e.mesh.geometry = g; e.changed = changedTime; }
        else {
          const m = new THREE.Mesh(g, self.occMat);
          m.matrixAutoUpdate = false;
          m.renderOrder = -10;
          self.group.add(m);
          e = { mesh:m, changed:changedTime };
          self.entries.set(item, e);
        }
      }
      const pose = frame.getPose(space, ref);
      if (pose) { e.mesh.visible = true; e.mesh.matrix.fromArray(pose.transform.matrix); }
      else e.mesh.visible = false;
    }
    if (frame.detectedPlanes) {
      frame.detectedPlanes.forEach(function (pl) {
        handle(pl, pl.planeSpace, self._planeGeometry.bind(self), pl.lastChangedTime);
      });
    }
    if (frame.detectedMeshes) {
      frame.detectedMeshes.forEach(function (xm) {
        handle(xm, xm.meshSpace, self._meshGeometry.bind(self), xm.lastChangedTime);
      });
    }
    this.entries.forEach(function (e, key) {
      if (!seen.has(key)) { self.group.remove(e.mesh); e.mesh.geometry.dispose(); self.entries.delete(key); }
    });
    if (S) S.occluderCount = this.entries.size;
  }
});

/* ============================================================
   swarm-player — generates formations parametrically (ported
   from the capstone's Python) instead of replaying a CSV.
   Selecting a formation captures wherever the swarm currently is
   and blends to the target using the SAME cosine S-curve the
   Python generator uses, over CONFIG.blendSeconds. While blending
   the state is explicitly TRANSITIONING (matching the real 8-class
   model: 7 formations + transitioning, not a forced top-1 guess).
   Once settled, the geometric classifier runs continuously as an
   independent check — same input/output contract a real trained
   classifier would fill.
   ============================================================ */
