/* tactical-grid.js — the dark-mode fallback ground grid (VR, not passthrough). */
AFRAME.registerComponent('tactical-grid', {
  init: function () {
    const g = new THREE.GridHelper(300,120, new THREE.Color('#1c6a4f'), new THREE.Color('#0c2f23'));
    g.material.transparent = true; g.material.opacity = 0.45; g.material.depthWrite = false;
    this.el.setObject3D('grid', g);
  }
});

/* ============================================================
   control-hub — the single dashboard for EVERYTHING.
   Opens on trigger/click, WORLD-LOCKED in front of the user so
   head-gaze can actually target its rows (a head-locked panel can
   never be gaze-targeted: it moves with the gaze, so the centre
   ray misses it identically forever — that was the old bug).
   Hidden entirely when closed.

   Input priority on trigger/click:
     hub open   -> row under reticle activates it, else close
     hub closed -> emitter under reticle claims it, else open hub
   ============================================================ */
