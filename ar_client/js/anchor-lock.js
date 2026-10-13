/* anchor-lock.js — WebXR anchor creation/tracking for the world root. */
AFRAME.registerComponent('anchor-lock', {
  init: function () {
    this.anchor = null; this.pending = false; this.recenterFlag = false;
    this.world = this.el.object3D; this._q = new THREE.Quaternion();
    this.lastPoseOK = false; this.anchorCount = 0;
    this.anchorPos = new THREE.Vector3();
  },
  requestRecenter: function () { this.recenterFlag = true; },
  bootstrap: function () {
    const scene = this.el.sceneEl, frame = scene.frame, r = scene.renderer;
    if (!frame || !r || !r.xr || !frame.createAnchor) return;
    const refSpace = r.xr.getReferenceSpace();
    if (!refSpace || this.anchor || this.pending) return;
    this._makeAnchor(frame, refSpace, new XRRigidTransform());
  },
  _makeAnchor: function (frame, refSpace, xform) {
    const self = this, old = this.anchor;
    this.pending = true;
    try {
      frame.createAnchor(xform, refSpace).then(function (a) {
        if (old && old.delete) { try { old.delete(); } catch (e) {} }
        self.anchor = a; self.pending = false; self.anchorCount++;
        if (window.RFX) window.RFX.anchored = true;
      }).catch(function () { self.pending = false; });
    } catch (e) { self.pending = false; }
  },
  tick: function () {
    const S = window.RFX;
    if (!S || !S.isAR) return;
    const scene = this.el.sceneEl, frame = scene.frame, r = scene.renderer;
    if (!frame || !r || !r.xr || !frame.createAnchor) return;
    const refSpace = r.xr.getReferenceSpace();
    if (!refSpace) return;
    if (this.recenterFlag) {
      this.recenterFlag = false;
      const vp = frame.getViewerPose(refSpace);
      if (vp) {
        const t = vp.transform;
        this._q.set(t.orientation.x, t.orientation.y, t.orientation.z, t.orientation.w);
        const e = new THREE.Euler().setFromQuaternion(this._q, 'YXZ');
        const yq = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, e.y, 0));
        this._makeAnchor(frame, refSpace, new XRRigidTransform(
          { x:t.position.x, y:0, z:t.position.z },
          { x:yq.x, y:yq.y, z:yq.z, w:yq.w }));
      }
    }
    if (!this.anchor && !this.pending) this._makeAnchor(frame, refSpace, new XRRigidTransform());
    if (this.anchor) {
      const pose = frame.getPose(this.anchor.anchorSpace, refSpace);
      this.lastPoseOK = !!pose;
      if (pose) {
        const p = pose.transform.position, o = pose.transform.orientation;
        this.world.position.set(p.x, p.y, p.z);
        this.world.quaternion.set(o.x, o.y, o.z, o.w);
        this.anchorPos.set(p.x, p.y, p.z);
      }
    } else {
      this.lastPoseOK = false;
    }
  }
});
