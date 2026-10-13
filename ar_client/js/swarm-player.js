/* swarm-player.js — drives the 6 swarm drones: formation selection, the pre-movement sweep + shape blend, converging shrink, and live classification. */
AFRAME.registerComponent('swarm-player', {
  init: function () {
    this.comps = [];
    this.formation = 'v_shape';
    this.pendingFormation = null;
    this.offsetsB = getFormationOffsets('v_shape', 1.0);
    this.offsetsA = this.offsetsB;
    this.blending = false; this.blendT = 0; this.settledAt = 0;
    this.currentLateral = [0, 0];   // actual live lateral (x,z) offset from anchor
    this.sweepAngle = 0;
    this.prevPts = null;
    this.lastClassify = 0;

    // classifier templates built from the EXACT ported geometry (no noise) —
    // dispersed/converging get a fixed reference draw purely for template
    // purposes; live dispersed/converging still redraws fresh each selection.
    const templates = {};
    FORMATION_NAMES.forEach(n => { templates[n] = getFormationOffsets(n, 1.0); });
    this.classifier = buildClassifier(templates);

    const self = this;
    this.el.addEventListener('targets-ready', function () {
      self.comps = Array.prototype.map.call(document.querySelectorAll('[rf-emitter]'),
        function (el) { return el.components['rf-emitter']; })
        .filter(function (c) { return c.e.mode === 'swarm'; });
      self.initOverlay();
    });
  },
  initOverlay: function () {
    const world = document.querySelector('#world').object3D;
    const geo = new THREE.BufferGeometry().setFromPoints(new Array(7).fill(new THREE.Vector3()));
    this.overlay = new THREE.LineLoop(geo, new THREE.LineBasicMaterial({
      color:0x39e0ff, transparent:true, opacity:0.55, depthTest:false, fog:false, toneMapped:false }));
    this.overlay.renderOrder = 11; world.add(this.overlay);
    this.centroidDot = new THREE.Mesh(new THREE.SphereGeometry(0.18,8,6),
      new THREE.MeshBasicMaterial({ color:0x39e0ff, transparent:true, opacity:0.7,
        depthTest:false, fog:false, toneMapped:false }));
    this.centroidDot.renderOrder = 11; world.add(this.centroidDot);
  },

  // Called by the control-hub FORMATION row (or any external caller).
  // Freezes wherever the swarm currently is — both shape AND lateral
  // position — as the new maneuver's start, so redirecting mid-transition
  // never causes a position jump.
  selectFormation: function (name) {
    if (name === this.formation && !this.blending) return;
    if (name === this.pendingFormation && this.blending) return;
    this.offsetsA = this.currentOffsets || this.offsetsB;
    this.offsetsB = getFormationOffsets(name, 1.0);
    this.pendingFormation = name;
    this.lateralCarry = this.currentLateral.slice();   // wherever the sweep actually is right now
    this.sweepAngle = Math.random() * 2 * Math.PI;      // fresh flourish direction for this maneuver
    this.blending = true; this.blendT = 0;
  },

  tick: function (time, dt) {
    if (!this.comps.length) return;
    const S = window.RFX;
    if (S.liveFeed) return;   // ros-feed.js owns pos/vel/classification while connected
    const stepped = S.playing ? Math.min(dt||16,50)/1000 * S.speed : 0;
    const now = time/1000;
    let offsets, alpha = 1;

    if (this.blending) {
      this.blendT += stepped;
      const total = CONFIG.sweepLeadSeconds + CONFIG.blendSeconds;
      const p = clamp(this.blendT / total, 0, 1);

      // shape only starts morphing AFTER the pre-movement lead — during the
      // lead the formation stays rigid, only translating, so the "dead stop"
      // is replaced by pure horizontal motion before any reconfiguring begins
      const shapeP = clamp((this.blendT - CONFIG.sweepLeadSeconds) / CONFIG.blendSeconds, 0, 1);
      alpha = cosineRamp(shapeP);
      offsets = blendOffsets(this.offsetsA, this.offsetsB, alpha);

      // lateral sweep = smoothly-decaying carry-over (wherever the sweep
      // actually was at the moment of selection) + a fresh flourish arc that
      // bumps out and back to zero. Both use zero-derivative-at-both-ends
      // curves, so redirecting mid-sweep never introduces a velocity kink.
      const carryEase = 1 - cosineRamp(p);                       // 1 -> 0, smooth
      const flourishMag = CONFIG.sweepAmplitude * Math.pow(Math.sin(Math.PI * p), 2);  // 0 -> peak -> 0
      const lx = this.lateralCarry[0]*carryEase + Math.cos(this.sweepAngle)*flourishMag;
      const lz = this.lateralCarry[1]*carryEase + Math.sin(this.sweepAngle)*flourishMag;
      this.currentLateral = [lx, lz];

      if (this.blendT >= total) {
        this.blending = false;
        this.formation = this.pendingFormation;
        this.pendingFormation = null;
        this.settledAt = now;
        this.currentLateral = [0, 0];
      }
    } else {
      let scale = 1.0;
      if (this.formation === 'converging') {
        const elapsed = now - this.settledAt;
        scale = 1.0 - 0.9 * clamp(elapsed / CONFIG.convergeShrinkSeconds, 0, 1);
      }
      offsets = this.offsetsB.map(p => [p[0]*scale, p[1]*scale, p[2]*scale]);
      this.currentLateral = [0, 0];
    }
    this.currentOffsets = offsets;

    // fixed anchor with a gentle vertical bob for life, plus the live
    // horizontal sweep offset computed above (zero once settled)
    const cx = this.currentLateral[0], cy = 100 + Math.sin(time*0.0004)*1.2, cz = this.currentLateral[1];
    const livePts = [];
    for (let d=0; d<6; d++) {
      const c = this.comps[d]; if (!c) continue;
      const e = c.e;
      const px = cx+offsets[d][0], py = cy+offsets[d][1], pz = cz+offsets[d][2];
      if (this.prevPts && stepped>0) {
        e.vel[0]=(px-this.prevPts[d][0])/stepped; e.vel[1]=(py-this.prevPts[d][1])/stepped; e.vel[2]=(pz-this.prevPts[d][2])/stepped;
      }
      e.pos[0]=px; e.pos[1]=py; e.pos[2]=pz;
      livePts.push([px,py,pz]);
    }
    this.prevPts = livePts;

    // classification state — TRANSITIONING is ground truth (we control the
    // blend), the geometric classifier only runs once settled, matching the
    // real model's own from -> transitioning -> to design.
    if (this.blending) {
      S.transState = 'transitioning';
      S.transFrom = this.formation; S.transTo = this.pendingFormation;
      S.transAlpha = clamp(this.blendT / (CONFIG.sweepLeadSeconds + CONFIG.blendSeconds), 0, 1);
      S.classLabel = null; S.classConf = null;
    } else {
      S.transState = 'settled';
      S.selectedFormation = this.formation;
      if (time - this.lastClassify > CONFIG.classifyEveryMs) {
        this.lastClassify = time;
        const result = this.classifier.classify(livePts);
        S.classLabel = result.label;
        S.classConf = result.confidences;
      }
    }

    if (this.overlay && S.showOverlay && S.mode === 'swarm') {
      this.overlay.visible = true; this.centroidDot.visible = true;
      const ordered = livePts.slice().sort((p,q)=>
        Math.atan2(p[2]-cz,p[0]-cx) - Math.atan2(q[2]-cz,q[0]-cx));
      const loopPts = ordered.map(p=>new THREE.Vector3(p[0],p[1],p[2]));
      loopPts.push(loopPts[0]);
      this.overlay.geometry.setFromPoints(loopPts);
      this.centroidDot.position.set(cx,cy,cz);
    } else if (this.overlay) {
      this.overlay.visible = false; this.centroidDot.visible = false;
    }
  }
});

/* ---------- classifier readout: TRANSITIONING (A -> B, blend%) while
   flying, CLASSIFIED + live confidence bars once settled. Leaner than
   before — no separate truth/match row, since "truth" is now just the
   user's own selection and restating it once matched adds nothing. ---------- */
