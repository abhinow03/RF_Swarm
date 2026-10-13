/* gaze-picker.js — head-gaze raycasting + claim logic for emitters. */
AFRAME.registerComponent('gaze-picker', {
  init: function () {
    const self = this;
    this.ray = new THREE.Raycaster(); this.ray.far = CONFIG.gazeFar;
    this.origin = new THREE.Vector3(); this.dir = new THREE.Vector3();
    this.current = null; this.candidate = null; this.since = 0;
    this.claimArmed = 0;
    this.byMode = { blobs:[], swarm:[] }; this.compsByMode = { blobs:[], swarm:[] };
    this.el.addEventListener('targets-ready', function () {
      const comps = Array.prototype.map.call(document.querySelectorAll('[rf-emitter]'),
        function (el) { return el.components['rf-emitter']; });
      self.compsByMode.blobs = comps.filter(function(c){ return c.e.mode==='blobs'; });
      self.compsByMode.swarm = comps.filter(function(c){ return c.e.mode==='swarm'; });
      self.byMode.blobs = self.compsByMode.blobs.map(function(c){ return c.orb; });
      self.byMode.swarm = self.compsByMode.swarm.map(function(c){ return c.orb; });
    });
    // input is owned by control-hub, which routes trigger/click to
    // instantClaim() only when the dashboard is closed.
  },
  // fresh raycast against whatever's under the reticle RIGHT NOW — click/
  // trigger acts immediately, it doesn't wait for the dwell timer in tick()
  // to have already picked something.
  pickTarget: function () {
    const cam = this.el.camera; const S = window.RFX;
    if (!cam || !S) return null;
    const orbs = this.byMode[S.mode], comps = this.compsByMode[S.mode];
    if (!orbs || !orbs.length) return null;
    cam.getWorldPosition(this.origin);
    cam.getWorldDirection(this.dir);
    this.ray.set(this.origin, this.dir);
    const hits = this.ray.intersectObjects(orbs, false);
    if (!hits.length) return null;
    const obj = hits[0].object;
    for (let i=0;i<comps.length;i++) if (comps[i].orb === obj) return comps[i];
    return null;
  },
  instantClaim: function () {
    const target = this.pickTarget();
    if (!target) return;
    if (this.current && this.current !== target) this.current.setFocused(false);
    target.setFocused(true); this.current = target; this.claimArmed = 0;
    this.claim();
  },
  claim: function () {
    if (!this.current) return;
    const on = this.current.toggleClaim();
    const hud = document.querySelector('#gaze-hud');
    if (hud && hud.components['gaze-hud']) hud.components['gaze-hud'].flash(on);
    this.claimArmed = 0;
  },
  tick: function (time, dt) {
    const cam = this.el.camera;
    const S = window.RFX;
    if (!cam || !S) return;
    // while the dashboard is open it owns the reticle — don't also target
    // emitters behind it, which would stack two panels on screen
    const hubEl = document.querySelector('#hub');
    const hub = hubEl && hubEl.components['control-hub'];
    if (hub && hub.open) {
      if (this.current) { this.current.setFocused(false); this.current = null; }
      S.gazed = null; S.claimProgress = 0;
      return;
    }
    const orbs = this.byMode[S.mode], comps = this.compsByMode[S.mode];
    if (this.current && this.current.e.mode !== S.mode) {
      this.current.setFocused(false); this.current = null; this.claimArmed = 0;
    }
    if (!orbs || !orbs.length) { window.RFX.gazed = null; window.RFX.claimProgress = 0; return; }
    cam.getWorldPosition(this.origin);
    cam.getWorldDirection(this.dir);
    this.ray.set(this.origin, this.dir);
    const hits = this.ray.intersectObjects(orbs, false);
    let comp = null;
    if (hits.length) {
      const obj = hits[0].object;
      for (let i=0;i<comps.length;i++) if (comps[i].orb === obj) { comp = comps[i]; break; }
    }
    if (comp !== this.candidate) { this.candidate = comp; this.since = time; this.claimArmed = 0; }
    if (comp && (time - this.since > CONFIG.dwellMs) && comp !== this.current) {
      if (this.current) this.current.setFocused(false);
      comp.setFocused(true); this.current = comp; this.claimArmed = 0;
    }
    if (!comp && this.current) { this.current.setFocused(false); this.current = null; this.claimArmed = 0; }
    if (this.current && comp === this.current) {
      this.claimArmed += (dt || 16);
      if (this.claimArmed > CONFIG.claimDwellMs + CONFIG.dwellMs) this.claim();
    }
    window.RFX.gazed = this.current;
    window.RFX.claimProgress = this.current
      ? Math.min(1, Math.max(0, (this.claimArmed - CONFIG.dwellMs) / CONFIG.claimDwellMs)) : 0;
  }
});
