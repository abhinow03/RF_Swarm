/* control-hub.js — the world-locked popup dashboard: all controls, joystick cursor, gaze+dwell and trigger input. */
AFRAME.registerComponent('control-hub', {
  init: function () {
    const self = this;
    this.open = false;
    this.panel = makeCanvasPanel(0.66, 0.50, 1800);
    this.el.object3D.add(this.panel.mesh);
    this.el.object3D.visible = false;

    const SCALE_PRESETS = [0.1, 0.05, 0.02, 0.008, 0.25];

    this.ROWS = [
      { key:'mode',     label:'MODE',      col:0, row:0, desc:'Switch demo: Swarm Formation flight vs Live Localisation emitter classes', get:function(S){ return [S.mode==='swarm'?'SWARM FORMATION':'LIVE LOCALISATION', '#39e0ff']; },
        run:function(S){ S.mode = S.mode==='swarm' ? 'blobs' : 'swarm'; } },
      { key:'formation',label:'FORMATION', col:0, row:1, desc:'Cycle the 7 formations. Swarm sweeps then morphs into the new shape', get:function(S){
          const sp = document.querySelector('a-scene').components['swarm-player'];
          const name = sp ? (sp.pendingFormation || sp.formation) : 'v_shape';
          return [name.toUpperCase().replace('_',' '), S.transState==='transitioning'?'#ffb454':'#39e0ff'];
        },
        run:function(S){
          const sp = document.querySelector('a-scene').components['swarm-player'];
          if (!sp) return;
          const cur = sp.pendingFormation || sp.formation;
          const idx = FORMATION_NAMES.indexOf(cur);
          sp.selectFormation(FORMATION_NAMES[(idx+1) % FORMATION_NAMES.length]);
          if (S.mode !== 'swarm') S.mode = 'swarm';
        } },
      { key:'playback', label:'PLAYBACK',  col:0, row:2, desc:'Freeze or resume all motion in the scene', get:function(S){ return [S.playing?'RUNNING':'PAUSED', S.playing?'#2dffa0':'#ffb454']; },
        run:function(S){ S.playing = !S.playing; } },
      { key:'speed',    label:'SPEED',     col:0, row:3, desc:'Time multiplier 0.5x - 4x, then wraps back to 0.5x', get:function(S){ return [S.speed+'x', '#39e0ff']; },
        run:function(S){ S.speed = S.speed>=4 ? 0.5 : S.speed*2; } },
      { key:'scale',    label:'SCALE',     col:0, row:4, desc:'World size. 1:10 fills a hall, 1:125 fits a desk. Ranges stay true', get:function(S){ return ['1:' + Math.round(1/S.scale), '#39e0ff']; },
        run:function(S){
          const i = SCALE_PRESETS.findIndex(v => Math.abs(v-S.scale) < 1e-6);
          S.scale = SCALE_PRESETS[(i+1) % SCALE_PRESETS.length];
        } },
      { key:'anchor',   label:'ANCHOR',    col:0, row:5, desc:'Re-pin the scene to where you stand now, facing your heading', get:function(){ return ['LOCK HERE', '#7fae9c']; },
        run:function(){ const w=document.querySelector('#world'); const c=w&&w.components['anchor-lock']; if(c) c.requestRecenter(); } },

      { key:'overlay',  label:'SHAPE OVR', col:1, row:0, desc:'Cyan outline joining the 6 drones - the geometry the classifier reads', get:function(S){ return [S.showOverlay?'ON':'OFF', S.showOverlay?'#2dffa0':'#7fae9c']; },
        run:function(S){ S.showOverlay = !S.showOverlay; } },
      { key:'trails',   label:'TRAILS',    col:1, row:1, desc:'Smoothed motion history curve behind every tracked object', get:function(S){ return [S.showTrails?'ON':'OFF', S.showTrails?'#2dffa0':'#7fae9c']; },
        run:function(S){ S.showTrails = !S.showTrails; } },
      { key:'debug',    label:'DEBUG',     col:1, row:2, desc:'Anchor lock state, tracking health, occluder count, head position', get:function(S){ return [S.showStatus?'ON':'OFF', S.showStatus?'#2dffa0':'#7fae9c']; },
        run:function(S){ S.showStatus = !S.showStatus; } },
      { key:'resetview',label:'RESET VIEW',col:1, row:3, desc:'Restore default scale and re-anchor in one action', get:function(){ return ['ANCHOR+SCALE', '#7fae9c']; },
        run:function(S){ S.scale = CONFIG.scale; const w=document.querySelector('#world'); const c=w&&w.components['anchor-lock']; if(c) c.requestRecenter(); } },
      { key:'exit',     label:'EXIT XR',   col:1, row:4, desc:'Leave the headset session and return to the flat page', get:function(S){ return [S.isAR?'LEAVE MR':'LEAVE VR', '#ff9640']; },
        run:function(){ const sc=document.querySelector('a-scene'); if (sc && sc.exitVR) sc.exitVR(); } },
      { key:'close',    label:'CLOSE',     col:1, row:5, desc:'Dismiss this dashboard. Trigger anywhere off-panel does the same', get:function(){ return ['DISMISS', '#ff5252']; },
        run:function(){ /* handled by activate(): closes the hub */ } }
    ];

    this.rowMeshes = this.ROWS.map(function (row) {
      const x = row.col===0 ? -0.165 : 0.165;
      const hp = new THREE.Mesh(new THREE.PlaneGeometry(0.30, 0.062),
        new THREE.MeshBasicMaterial({ visible:false }));
      hp.position.set(x, self.rowY(row.row), 0.004);
      self.panel.mesh.add(hp);
      return hp;
    });

    // full-panel backing plane. The reticle deliberately rests in the gap
    // between columns when the hub opens (so the dwell timer can't auto-fire
    // a row the instant it appears) — this backing lets us tell "aimed at the
    // panel but between rows" (do nothing) apart from "aimed away from the
    // panel" (dismiss), so opening then triggering doesn't instantly close it.
    this.bgMesh = new THREE.Mesh(new THREE.PlaneGeometry(0.66, 0.50),
      new THREE.MeshBasicMaterial({ visible:false }));
    this.bgMesh.position.set(0, 0, 0.001);
    this.panel.mesh.add(this.bgMesh);

    // ---- joystick pointer ----
    // Thumbstick drives a cursor across the panel in panel-local metres.
    // Far more precise than head-gaze for a dense 12-row grid, and it means
    // the user's head can stay still while selecting.
    this.cursor = new THREE.Group();
    const cRing = new THREE.Mesh(new THREE.RingGeometry(0.011, 0.017, 24),
      new THREE.MeshBasicMaterial({ color:0xffffff, transparent:false,
        fog:false, toneMapped:false, depthTest:false, side:THREE.DoubleSide }));
    cRing.renderOrder = 1001;
    const cDot = new THREE.Mesh(new THREE.CircleGeometry(0.005, 16),
      new THREE.MeshBasicMaterial({ color:0x39e0ff, transparent:false,
        fog:false, toneMapped:false, depthTest:false }));
    cDot.renderOrder = 1002;
    this.cursor.add(cRing); this.cursor.add(cDot);
    this.cursor.position.set(0, 0, 0.012);
    this.panel.mesh.add(this.cursor);

    this.cursorPos = { x:0, y:0 };
    this.stick = { x:0, y:0 };
    this.usedStick = false;

    const onStick = function (e) {
      if (!e.detail) return;
      const x = (e.detail.x !== undefined) ? e.detail.x : 0;
      const y = (e.detail.y !== undefined) ? e.detail.y : 0;
      self.stick.x = x; self.stick.y = y;
      if (Math.abs(x) > 0.15 || Math.abs(y) > 0.15) self.usedStick = true;
    };
    this.el.sceneEl.addEventListener('thumbstickmoved', onStick);
    // raw fallback for controllers whose profile doesn't map thumbstickmoved
    this.el.sceneEl.addEventListener('axismove', function (e) {
      if (!e.detail || !e.detail.axis || e.detail.axis.length < 2) return;
      const a = e.detail.axis;
      const x = a.length >= 4 ? a[2] : a[0];
      const y = a.length >= 4 ? a[3] : a[1];
      self.stick.x = x; self.stick.y = y;
      if (Math.abs(x) > 0.15 || Math.abs(y) > 0.15) self.usedStick = true;
    });

    this.ray = new THREE.Raycaster();
    this.origin = new THREE.Vector3(); this.dir = new THREE.Vector3();
    this.candidate = -1; this.since = 0; this.armed = 0;
    this.flashRow = -1; this.flashT = 0;

    // single unified input path: controller trigger in XR, mouse/touch on flat
    const handler = function () { self.onTrigger(); };
    this.el.sceneEl.addEventListener('triggerdown', handler);
    window.addEventListener('click', function (e) {
      if (e.target && e.target.tagName === 'CANVAS') handler();
    });

    this.render();
  },
  rowY: function (i) { return 0.13 - i*0.062; },

  // place the panel in world space, upright, centred in front of the viewer
  openAt: function () {
    const cam = this.el.sceneEl.camera;
    if (!cam) return;
    const camPos = new THREE.Vector3(); cam.getWorldPosition(camPos);
    const dir = new THREE.Vector3(); cam.getWorldDirection(dir);
    const flat = new THREE.Vector3(dir.x, 0, dir.z);
    if (flat.lengthSq() < 1e-6) flat.set(0,0,-1);
    flat.normalize();
    const o = this.el.object3D;
    o.position.copy(camPos).add(flat.multiplyScalar(0.95));
    o.position.y = camPos.y;          // eye level, dead centre of view
    o.lookAt(camPos);                 // PlaneGeometry front face is +Z
    o.visible = true;
    this.open = true;
    this.candidate = -1; this.armed = 0;
    // cursor starts dead-centre, which sits in the gap between columns so
    // the dwell timer can't auto-fire a row the instant the hub appears
    this.cursorPos.x = 0; this.cursorPos.y = 0;
    this.usedStick = false;
    this.setLaserVisible(false);
    this.render();
  },
  close: function () {
    this.open = false;
    this.el.object3D.visible = false;
    this.candidate = -1; this.armed = 0;
    this.setLaserVisible(true);
  },
  // hide the controller ray while the dashboard owns input, restore after
  setLaserVisible: function (vis) {
    const rays = document.querySelectorAll('[laser-controls]');
    for (let i=0;i<rays.length;i++) {
      try { rays[i].setAttribute('raycaster', 'showLine', vis); } catch (e) {}
      const lineObj = rays[i].getObject3D('line');
      if (lineObj) lineObj.visible = vis;
    }
  },
  // which row is under the joystick cursor (panel-local geometry, no raycast)
  cursorRow: function () {
    for (let i=0;i<this.ROWS.length;i++) {
      const row = this.ROWS[i];
      const cx = row.col===0 ? -0.165 : 0.165;
      const cy = this.rowY(row.row);
      if (Math.abs(this.cursorPos.x - cx) <= 0.15 &&
          Math.abs(this.cursorPos.y - cy) <= 0.031) return i;
    }
    return -1;
  },
  // joystick takes over once touched; head-gaze remains the fallback so the
  // hub is still usable if the stick never reports (unknown controller profile)
  hoveredRow: function () {
    return this.usedStick ? this.cursorRow() : this.pickRow();
  },

  pickRow: function () {
    const cam = this.el.sceneEl.camera;
    if (!cam || !this.open) return -1;
    cam.getWorldPosition(this.origin);
    cam.getWorldDirection(this.dir);
    this.ray.set(this.origin, this.dir);
    const hits = this.ray.intersectObjects(this.rowMeshes, false);
    return hits.length ? this.rowMeshes.indexOf(hits[0].object) : -1;
  },
  pickPanel: function () {
    const cam = this.el.sceneEl.camera;
    if (!cam || !this.open) return false;
    cam.getWorldPosition(this.origin);
    cam.getWorldDirection(this.dir);
    this.ray.set(this.origin, this.dir);
    return this.ray.intersectObject(this.bgMesh, false).length > 0;
  },

  onTrigger: function () {
    // selectstart and A-Frame's triggerdown can both fire for one pull;
    // collapse them into a single action.
    const nowMs = performance.now();
    if (this._lastTrigger && nowMs - this._lastTrigger < 250) return;
    this._lastTrigger = nowMs;
    if (this.open) {
      const idx = this.hoveredRow();
      if (idx >= 0) { this.activate(idx); return; }
      // joystick mode: cursor is always on the panel, so use the CLOSE row.
      // gaze mode: aiming away from the panel dismisses it.
      if (this.usedStick || this.pickPanel()) return;
      this.close();
      return;
    }
    // hub closed: prefer claiming an emitter under the reticle, else open hub
    const gp = this.el.sceneEl.components['gaze-picker'];
    if (gp && gp.pickTarget && gp.pickTarget()) { gp.instantClaim(); return; }
    this.openAt();
  },

  activate: function (idx) {
    const S = window.RFX;
    if (!S) return;
    const row = this.ROWS[idx];
    row.run(S);
    if (row.key === 'close') { this.close(); return; }
    this.flashRow = idx; this.flashT = 260;
    this.armed = 0; this.candidate = -1;
    this.render();
  },

  tick: function (time, dt) {
    const S = window.RFX;
    if (!S || !this.open) return;
    if (this.flashT > 0) { this.flashT -= (dt||16); if (this.flashT<=0) { this.flashRow=-1; this.render(); } }

    // integrate thumbstick into cursor position (event only fires on change,
    // so holding the stick needs per-frame integration)
    const dts = Math.min(dt||16, 50)/1000;
    if (Math.abs(this.stick.x) > 0.15 || Math.abs(this.stick.y) > 0.15) {
      this.cursorPos.x = clamp(this.cursorPos.x + this.stick.x * CONFIG.cursorSpeed * dts, -0.32, 0.32);
      this.cursorPos.y = clamp(this.cursorPos.y + this.stick.y * CONFIG.cursorSpeed * dts, -0.24, 0.24);
    }
    this.cursor.visible = this.usedStick;
    this.cursor.position.set(this.cursorPos.x, this.cursorPos.y, 0.012);

    const gazeIdx = this.hoveredRow();
    if (gazeIdx !== this.candidate) { this.candidate = gazeIdx; this.since = time; this.armed = 0; }
    let progress = 0;
    if (gazeIdx >= 0) {
      this.armed = time - this.since;
      progress = clamp(this.armed / CONFIG.uiDwellMs, 0, 1);
      if (this.armed > CONFIG.uiDwellMs) { this.activate(gazeIdx); return; }
    }
    if (gazeIdx >= 0 && (!this._lastDraw || time - this._lastDraw > 90)) {
      this._lastDraw = time; this.render(gazeIdx, progress);
    } else if (gazeIdx < 0 && this._prevGazeIdx >= 0) {
      this.render();
    }
    this._prevGazeIdx = gazeIdx;
    if (!this._lastFull || time - this._lastFull > 400) { this._lastFull = time; this.render(gazeIdx, progress); }
  },

  render: function (hoverIdx, hoverProg) {
    const S = window.RFX; if (!S) return;
    const ctx = this.panel.ctx, cv = this.panel.cv, W = cv.width, H = cv.height;
    const PW = this.panel.w, PH = this.panel.h;
    ctx.clearRect(0,0,W,H);
    ctx.fillStyle = 'rgba(2,10,7,0.95)';
    roundRect(ctx,4,4,W-8,H-8,20); ctx.fill();
    ctx.strokeStyle = '#2dffa0'; ctx.lineWidth = 5; ctx.stroke();

    ctx.textBaseline='middle'; ctx.textAlign='center';
    ctx.fillStyle = '#2dffa0';
    ctx.font = 'bold ' + Math.round(H*0.055) + 'px ui-monospace, monospace';
    ctx.fillText('CONTROL HUB', W/2, H*0.065);
    ctx.fillStyle = '#7fae9c';
    ctx.font = Math.round(H*0.032) + 'px ui-monospace, monospace';
    ctx.fillText('gaze a row and hold, or pull trigger \u00b7 trigger off-panel to close', W/2, H*0.115);

    ctx.strokeStyle = 'rgba(45,255,160,0.18)'; ctx.lineWidth=2;
    ctx.beginPath(); ctx.moveTo(W/2, H*0.16); ctx.lineTo(W/2, H*0.84); ctx.stroke();

    const cellW = 0.30/PW * W, cellH = 0.062/PH * H;
    for (let i=0;i<this.ROWS.length;i++){
      const row = this.ROWS[i];
      const xm = row.col===0 ? -0.165 : 0.165;
      const ym = this.rowY(row.row);
      const cx = W*(0.5 + xm/PW), cy = H*(0.5 - ym/PH);
      const isHover = i === hoverIdx;
      const isFlash = i === this.flashRow;

      if (isFlash) {
        ctx.fillStyle = 'rgba(255,180,84,0.35)';
        roundRect(ctx, cx-cellW/2, cy-cellH/2, cellW, cellH, 8); ctx.fill();
      } else if (isHover) {
        ctx.fillStyle = 'rgba(45,255,160,0.10)';
        roundRect(ctx, cx-cellW/2, cy-cellH/2, cellW, cellH, 8); ctx.fill();
        if (hoverProg > 0.01) {
          ctx.fillStyle = 'rgba(45,255,160,0.32)';
          roundRect(ctx, cx-cellW/2, cy-cellH/2, cellW*hoverProg, cellH, 8); ctx.fill();
        }
      }
      const val = row.get(S);
      ctx.textAlign='left'; ctx.fillStyle = isHover ? '#eafff6' : '#cfe6dc';
      ctx.font = 'bold ' + Math.round(H*0.030) + 'px ui-monospace, monospace';
      ctx.fillText(row.label, cx-cellW/2+8, cy-cellH*0.18);
      ctx.fillStyle = val[1];
      ctx.font = Math.round(H*0.028) + 'px ui-monospace, monospace';
      ctx.fillText(val[0], cx-cellW/2+8, cy+cellH*0.26);
    }

    const fy = H*0.90;
    ctx.strokeStyle = 'rgba(45,255,160,0.25)'; ctx.lineWidth=2;
    ctx.beginPath(); ctx.moveTo(W*0.06,fy-H*0.025); ctx.lineTo(W*0.94,fy-H*0.025); ctx.stroke();
    // hovering a row explains that row; otherwise fall back to status/debug
    if (hoverIdx >= 0 && this.ROWS[hoverIdx].desc) {
      ctx.textAlign='left'; ctx.fillStyle='#39e0ff';
      ctx.font = 'bold ' + Math.round(H*0.030) + 'px ui-monospace, monospace';
      ctx.fillText(this.ROWS[hoverIdx].label, W*0.06, fy+H*0.020);
      ctx.fillStyle='#cfe6dc';
      ctx.font = Math.round(H*0.026) + 'px ui-monospace, monospace';
      const words = this.ROWS[hoverIdx].desc.split(' ');
      const maxW = W*0.88; let line='', y=fy+H*0.055;
      for (let w=0; w<words.length; w++) {
        const test = line ? line+' '+words[w] : words[w];
        if (ctx.measureText(test).width > maxW && line) {
          ctx.fillText(line, W*0.06, y); line = words[w]; y += H*0.032;
        } else line = test;
      }
      if (line) ctx.fillText(line, W*0.06, y);
    } else if (S.showStatus || this._forceStatus) {
      const st = this.readStatus();
      ctx.textAlign='left'; ctx.fillStyle = st.color;
      ctx.font = 'bold ' + Math.round(H*0.030) + 'px ui-monospace, monospace';
      ctx.fillText(st.line1, W*0.06, fy+H*0.020);
      ctx.fillStyle = '#7fae9c';
      ctx.font = Math.round(H*0.026) + 'px ui-monospace, monospace';
      ctx.fillText(st.line2, W*0.06, fy+H*0.055);
    } else {
      ctx.textAlign='center'; ctx.fillStyle = '#3a5b50';
      ctx.font = Math.round(H*0.026) + 'px ui-monospace, monospace';
      ctx.fillText('debug info hidden', W/2, fy+H*0.035);
    }
    this.panel.tex.needsUpdate = true;
  },

  readStatus: function () {
    const S = window.RFX;
    const worldEl = document.querySelector('#world');
    const al = worldEl && worldEl.components['anchor-lock'];
    let anchor='OFF', track='\u2014', hx='--', hz='--', occ=0;
    const cam = this.el.sceneEl.camera;
    if (cam && worldEl) {
      const v = new THREE.Vector3(); cam.getWorldPosition(v);
      v.sub(worldEl.object3D.position);
      const iq = worldEl.object3D.quaternion.clone().invert();
      v.applyQuaternion(iq);
      hx = v.x.toFixed(2); hz = v.z.toFixed(2);
    }
    if (al) { anchor = al.anchor ? 'LOCKED' : (S.isAR ? 'PENDING' : 'OFF'); track = al.lastPoseOK ? 'OK' : (al.anchor?'LOST':'\u2014'); }
    occ = S.occluderCount || 0;
    this._forceStatus = (anchor === 'PENDING');
    const good = anchor==='LOCKED' && track==='OK';
    return {
      color: good ? '#2dffa0' : (anchor==='PENDING' ? '#ffb454' : '#ff5252'),
      line1: 'ANCHOR ' + anchor + '   TRK ' + track + '   OCC ' + occ,
      line2: 'HEAD ' + hx + '/' + hz + 'm'
    };
  }
});
