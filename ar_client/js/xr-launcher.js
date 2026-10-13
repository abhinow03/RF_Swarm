/* xr-launcher.js — WebXR session management (AR/VR entry, passthrough). */
AFRAME.registerComponent('xr-launcher', {
  init: function () {
    const self = this;
    this.renderer = null;
    this.bgc = new THREE.Color(CONFIG.bg);
    this.fog = new THREE.FogExp2(this.bgc.getHex(), CONFIG.fogDensity);
    this.arBtn = document.getElementById('btn-ar');
    this.vrBtn = document.getElementById('btn-vr');
    this.diag  = document.getElementById('diag');
    this.setWorld(true);
    const ready = () => { self.renderer = self.el.renderer; self.probe(); };
    if (this.el.renderer) ready(); else this.el.addEventListener('loaded', ready);
    this.arBtn.addEventListener('click', () => self.enter('immersive-ar'));
    this.vrBtn.addEventListener('click', () => self.enter('immersive-vr'));
    this.el.addEventListener('exit-vr', () => {
      document.body.classList.remove('in-xr');
      if (window.RFX) window.RFX.isAR = false;
      self.setWorld(true);
    });
  },
  probe: function () {
    const self = this;
    if (!navigator.xr) {
      this.diag.innerHTML = 'navigator.xr MISSING — not a WebXR browser / insecure origin';
      this.diag.className = 'bad'; return;
    }
    Promise.all([
      navigator.xr.isSessionSupported('immersive-ar').catch(()=>false),
      navigator.xr.isSessionSupported('immersive-vr').catch(()=>false)
    ]).then(([ar, vr]) => {
      self.arBtn.disabled = !ar; self.vrBtn.disabled = !vr;
      self.arBtn.classList.toggle('on', ar); self.vrBtn.classList.toggle('on', vr);
      self.diag.innerHTML =
        'immersive-ar: <b>' + (ar?'YES':'NO') + '</b> &nbsp; immersive-vr: <b>' + (vr?'YES':'NO') + '</b>' +
        (ar ? '<br>tap ENTER MIXED REALITY for passthrough'
            : '<br>no passthrough — use the standalone Meta Quest Browser (not Link)');
      self.diag.className = ar ? 'ok' : 'warn';
    });
  },
  enter: function (mode) {
    const self = this;
    if (!navigator.xr || !this.renderer) return;
    const isAR = mode === 'immersive-ar';
    const opts = isAR
      ? { requiredFeatures:['local-floor'], optionalFeatures:['bounded-floor','hand-tracking','anchors','plane-detection','mesh-detection'] }
      : { requiredFeatures:['local-floor'], optionalFeatures:['bounded-floor','hand-tracking'] };
    navigator.xr.requestSession(mode, opts).then((session) => {
      self.setWorld(!isAR);
      if (window.RFX) {
        window.RFX.isAR = isAR;
        window.RFX.arEnterTime = performance.now();
      }
      self.renderer.xr.setReferenceSpaceType('local-floor');
      // Route the WebXR-native select event straight to the dashboard. This
      // is far more reliable than depending on A-Frame's laser-controls
      // emitting 'triggerdown' (which needs the right gamepad profile to be
      // recognised), and it also makes hand-tracking pinch work for free.
      session.addEventListener('selectstart', function () {
        const hubEl = document.querySelector('#hub');
        const hub = hubEl && hubEl.components['control-hub'];
        if (hub) hub.onTrigger();
      });
      self.renderer.xr.setSession(session).then(() => {
        document.body.classList.add('in-xr');
        if (isAR) {
          const world = document.querySelector('#world');
          const al = world && world.components['anchor-lock'];
          if (al) requestAnimationFrame(() => requestAnimationFrame(() => al.bootstrap()));
        }
      });
      session.addEventListener('end', () => {
        document.body.classList.remove('in-xr');
        if (window.RFX) window.RFX.isAR = false;
        self.setWorld(true);
      });
    }).catch((err) => {
      self.diag.innerHTML = 'session failed: ' + (err && err.message ? err.message : err);
      self.diag.className = 'bad';
    });
  },
  setWorld: function (isDark) {
    const g = this.el.querySelector('#ground'), gr = this.el.querySelector('#grid');
    if (g)  g.setAttribute('visible', isDark);
    if (gr) gr.setAttribute('visible', isDark);
    this.el.object3D.background = isDark ? this.bgc : null;
    this.el.object3D.fog        = isDark ? this.fog : null;
    const r = this.el.renderer;
    if (r) { r.setClearColor(isDark ? this.bgc.getHex() : 0x000000, isDark ? 1 : 0); r.setClearAlpha(isDark ? 1 : 0); }
    document.body.style.background = isDark ? CONFIG.bg : 'transparent';
  }
});
