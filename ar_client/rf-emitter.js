/* rf-emitter.js — the per-emitter beacon: model, beam, occlusion ghost,
   claim state, breadcrumb trail. Includes its sole-consumer helpers
   (model builder, wireframe clone, beam shader) rather than utils.js. */
function makeBeamMaterial(colorHex) {
  return new THREE.ShaderMaterial({
    transparent:true, depthWrite:false, blending:THREE.NormalBlending, side:THREE.DoubleSide,
    uniforms: {
      uColor:{ value:new THREE.Color(colorHex) }, uTime:{ value:0 },
      uActive:{ value:0.0 }, uOpacity:{ value:0.9 }
    },
    vertexShader: [
      'varying vec2 vUv;',
      'void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }'
    ].join('\n'),
    fragmentShader: [
      'uniform vec3 uColor; uniform float uTime; uniform float uActive; uniform float uOpacity;',
      'varying vec2 vUv;',
      'void main(){',
      '  float fade = pow(1.0 - vUv.y, 1.5);',
      '  float bands = sin((vUv.y * 20.0) - uTime * 2.4);',
      '  bands = smoothstep(0.5, 1.0, bands) * 0.6;',
      '  float sweep = smoothstep(0.88, 1.0, sin((vUv.y * 3.0) - uTime * (2.0 + uActive * 5.0)));',
      '  float edge = sin(vUv.x * 3.14159);',
      '  float a = (fade * (0.48 + bands + sweep * 0.8)) * edge;',
      '  a *= uOpacity * (0.7 + uActive * 0.8);',
      '  gl_FragColor = vec4(uColor * (1.0 + uActive * 0.7), a);',
      '}'
    ].join('\n')
  });
}

// One shared, lazily-built tileable carbon-weave texture, reused across
// every body material (built once, not per-instance — cheap in memory and
// GPU state). A fine diagonal crosshatch in dark-on-darker grey; enough to
// break up the "flat plastic" read at close range without needing a real
// texture asset file.
let _carbonTex = null;
function getCarbonWeaveTexture() {
  if (_carbonTex) return _carbonTex;
  const size = 128;
  const cv = document.createElement('canvas');
  cv.width = size; cv.height = size;
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#141517'; ctx.fillRect(0, 0, size, size);
  ctx.strokeStyle = 'rgba(60,64,70,0.55)'; ctx.lineWidth = 1.4;
  const step = 8;
  for (let i = -size; i < size*2; i += step) {
    ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i+size, size); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(i, size); ctx.lineTo(i+size, 0); ctx.stroke();
  }
  ctx.fillStyle = 'rgba(0,0,0,0.18)';
  for (let i=0;i<40;i++) ctx.fillRect(Math.random()*size, Math.random()*size, 2, 2);
  const tex = new THREE.CanvasTexture(cv);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(3, 3);
  _carbonTex = tex;
  return tex;
}

// Confidence [0,1] -> emissive intensity. Kept deliberately dim/narrow —
// this is a small status indicator, not a beacon: a low-confidence read
// should look like a barely-lit LED, a high-confidence read a solid,
// steady one, never a glow that competes with the hardware silhouette.
function confidenceToGlow(confidence) {
  const c = (typeof confidence === 'number' && isFinite(confidence)) ? clamp(confidence, 0, 1) : 0.7;
  return 0.35 + c * 0.85;   // range ~0.35 (barely lit) .. 1.2 (solid, still not blown out)
}

function buildEmitterModel(shape, color, confidence) {
  const g = new THREE.Group();
  const bodyMats = [], accentMats = [];
  const glow = confidenceToGlow(confidence);
  const carbonTex = getCarbonWeaveTexture();

  // Gunmetal/carbon chassis. Lower metalness than a pure-metal look would
  // want (there's no environment map in this scene, so high metalness with
  // nothing to reflect just goes flat dark) — the carbon-weave map supplies
  // the surface detail instead of relying on specular alone.
  const BODY = () => {
    const m = new THREE.MeshStandardMaterial({
      color: 0x1b1d20, map: carbonTex, roughness: 0.55, metalness: 0.35,
      flatShading: true, fog: false, toneMapped: false
    });
    bodyMats.push(m); return m;
  };
  // Bright bare-metal hardware: rotor hubs, screws, joints. Non-glowing —
  // this is what reads as "small functional metal part", not an indicator.
  const TRIM = () => new THREE.MeshStandardMaterial({
    color: 0x9aa1ab, roughness: 0.28, metalness: 0.85, flatShading: true,
    fog: false, toneMapped: false
  });
  // Small glowing status indicator — deliberately tiny geometry (a strip or
  // dot), desaturated slightly from the raw class colour so it reads as a
  // status LED against dark hardware rather than a coloured surface.
  const ACCENT = () => {
    const c = color.clone().multiplyScalar(0.88);
    const m = new THREE.MeshStandardMaterial({
      color: c.clone().multiplyScalar(0.12), emissive: c,
      emissiveIntensity: glow, roughness: 0.5, metalness: 0.1, flatShading: true,
      fog: false, toneMapped: false
    });
    accentMats.push(m); return m;
  };

  if (shape === 'drone') {
    // compact rectangular avionics housing, not a symmetric puck
    const housing = new THREE.Mesh(new THREE.BoxGeometry(0.30, 0.11, 0.46), BODY());
    g.add(housing);
    // small raised sensor/GPS bump for silhouette detail
    const bump = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.05, 0.14), BODY());
    bump.position.set(0, 0.08, -0.08); g.add(bump);
    // corner screws — tiny bright-metal dots, hardware realism
    [[0.12,0.19],[0.12,-0.19],[-0.12,0.19],[-0.12,-0.19]].forEach(function(p){
      const s = new THREE.Mesh(new THREE.SphereGeometry(0.014,5,4), TRIM());
      s.position.set(p[0],0.058,p[1]); g.add(s);
    });
    // slim motor arms — structural, same carbon finish as the body
    const armG = new THREE.CylinderGeometry(0.020, 0.020, 0.62, 6);
    for (let i=0;i<2;i++){
      const arm = new THREE.Mesh(armG, BODY());
      arm.rotation.z = Math.PI/2; arm.rotation.y = (i===0?1:-1)*Math.PI/4;
      g.add(arm);
    }
    // small functional rotor hubs — bare metal, NOT glowing rings
    [[0.32,0.32],[0.32,-0.32],[-0.32,0.32],[-0.32,-0.32]].forEach(function(p){
      const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.055,0.06,0.045,10), TRIM());
      hub.position.set(p[0],0.055,p[1]); g.add(hub);
      // two thin blades per rotor — cheap, reads as an actual propeller
      for (let bIdx=0; bIdx<2; bIdx++){
        const blade = new THREE.Mesh(new THREE.BoxGeometry(0.18,0.006,0.028), BODY());
        blade.position.set(p[0],0.075,p[1]); blade.rotation.y = bIdx*Math.PI/2 + Math.PI/6;
        g.add(blade);
      }
    });
    // ONE small status strip — the only glowing element on the whole drone
    const strip = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.014, 0.20), ACCENT());
    strip.position.set(0, 0.065, 0.10); g.add(strip);
  } else if (shape === 'remote') {
    g.add(new THREE.Mesh(new THREE.BoxGeometry(0.5,0.14,0.75), BODY()));
    [[0.20,0.05],[-0.20,0.05],[0.20,-0.05],[-0.20,-0.05]].forEach(function(p){
      const s = new THREE.Mesh(new THREE.SphereGeometry(0.013,5,4), TRIM());
      s.position.set(p[0],0.075,p[1]); g.add(s);
    });
    const antG = new THREE.CylinderGeometry(0.018,0.018,0.55,6);
    [-0.16,0.16].forEach(function(x,i){
      const a = new THREE.Mesh(antG, BODY());
      a.position.set(x,0.32,-0.28); a.rotation.x = -0.5; a.rotation.z = i===0?0.25:-0.25;
      g.add(a);
      // small transmit-indicator dot, not a bright ball
      const tip = new THREE.Mesh(new THREE.SphereGeometry(0.028,6,5), ACCENT());
      tip.position.set(x + (i===0?0.13:-0.13), 0.55, -0.53); g.add(tip);
    });
    [-0.13,0.13].forEach(function(x){
      const stick = new THREE.Mesh(new THREE.SphereGeometry(0.05,8,6), TRIM());
      stick.position.set(x,0.11,0.16); g.add(stick);
    });
  } else if (shape === 'antenna') {
    const base = new THREE.Mesh(new THREE.BoxGeometry(0.36,0.13,0.36), BODY());
    base.position.y = -0.35; g.add(base);
    [[0.13,0.13],[0.13,-0.13],[-0.13,0.13],[-0.13,-0.13]].forEach(function(p){
      const s = new THREE.Mesh(new THREE.SphereGeometry(0.012,5,4), TRIM());
      s.position.set(p[0],-0.29,p[1]); g.add(s);
    });
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.028,0.038,0.8,6), BODY());
    pole.position.y = 0.05; g.add(pole);
    const collar = new THREE.Mesh(new THREE.CylinderGeometry(0.05,0.05,0.03,8), TRIM());
    collar.position.y = -0.30; g.add(collar);
    // small transmit tip, two thin signal-wave rings (down from three, thinner, dimmer)
    const tip = new THREE.Mesh(new THREE.SphereGeometry(0.04,8,6), ACCENT());
    tip.position.y = 0.48; g.add(tip);
    [0.26,0.42].forEach(function(r){
      const arc = new THREE.Mesh(new THREE.TorusGeometry(r,0.012,6,14,Math.PI*0.55), ACCENT());
      arc.position.y = 0.48; arc.rotation.z = Math.PI*0.225;
      g.add(arc);
    });
  } else {
    // general floater: faceted sensor-pod body, thin bright metal trim,
    // one narrow glowing band rather than a broad equatorial glow
    g.add(new THREE.Mesh(new THREE.IcosahedronGeometry(0.5, 1), BODY()));
    const trim = new THREE.Mesh(new THREE.TorusGeometry(0.58,0.010,4,24), TRIM());
    trim.rotation.x = Math.PI/2; trim.rotation.z = Math.PI/3; g.add(trim);
    const band = new THREE.Mesh(new THREE.TorusGeometry(0.68,0.014,6,24), ACCENT());
    band.rotation.x = Math.PI/2; g.add(band);
  }
  g.userData.bodyMats = bodyMats;
  g.userData.accentMats = accentMats;
  g.userData.mats = accentMats;   // legacy alias: recolor-on-claim touches
                                   // only the small indicator, never the chassis
  return g;
}

function makeWireframeClone(src, color) {
  const g = new THREE.Group(); const mats = [];
  src.traverse(function (o) {
    if (o.isMesh) {
      const m = new THREE.Mesh(o.geometry,
        new THREE.MeshBasicMaterial({ color:color.clone(), wireframe:true, transparent:true,
          opacity:0.6, depthTest:false, fog:false, toneMapped:false }));
      m.position.copy(o.position); m.rotation.copy(o.rotation); m.scale.copy(o.scale);
      m.renderOrder = 12; mats.push(m.material); g.add(m);
    }
  });
  g.userData.mats = mats;
  return g;
}

AFRAME.registerComponent('rf-emitter', {
  schema: { idx:{type:'int', default:0} },
  init: function () {
    const e = this.e = ALL_EMITTERS[this.data.idx];
    this.baseColor = e.colorOverride || CLASSES[e.cls].color;
    this.claimed = false;
    const root = this.el.object3D;
    const c3 = new THREE.Color(this.baseColor);

    // All beacon geometry uses NORMAL blending, never additive: additive adds
    // to the passthrough behind it, so it washes out toward white against a
    // bright wall. Normal blending composites properly and holds its colour.
    this.beamMat = makeBeamMaterial(this.baseColor);
    this.beam = new THREE.Mesh(new THREE.CylinderGeometry(0.34,0.34,30,12,1,true), this.beamMat);
    this.beam.position.y = 15; root.add(this.beam);

    const soft = (o) => new THREE.MeshBasicMaterial({
      color:c3, transparent:true, opacity:o, fog:false, depthWrite:false,
      toneMapped:false, blending:THREE.NormalBlending });
    const solid = () => new THREE.MeshBasicMaterial({
      color:c3, transparent:false, opacity:1, fog:false, toneMapped:false });

    this.spine = new THREE.Mesh(new THREE.CylinderGeometry(0.10,0.10,30,8), solid());
    this.spine.position.y = 15; root.add(this.spine);

    // dark opaque outline ring under the bright ring: pure contrast device,
    // gives the locator a hard edge on light-coloured floors
    this.underRing = new THREE.Mesh(new THREE.RingGeometry(1.55,2.35,48),
      new THREE.MeshBasicMaterial({ color:0x02100a, transparent:false,
        fog:false, toneMapped:false, depthTest:false, side:THREE.DoubleSide }));
    this.underRing.renderOrder = 7;
    this.underRing.rotation.x = -Math.PI/2; this.underRing.position.y = 0.02; root.add(this.underRing);

    this.ring = new THREE.Mesh(new THREE.RingGeometry(1.7,2.2,48), solid());
    this.ring.material.side = THREE.DoubleSide;
    this.ring.material.depthTest = false; this.ring.renderOrder = 8;
    this.ring.rotation.x = -Math.PI/2; this.ring.position.y = 0.03; root.add(this.ring);
    this.pulse = new THREE.Mesh(new THREE.RingGeometry(1.2,1.5,48), soft(0.85));
    this.pulse.material.side = THREE.DoubleSide;
    this.pulse.material.depthTest = false; this.pulse.renderOrder = 8;
    this.pulse.rotation.x = -Math.PI/2; this.pulse.position.y = 0.05; root.add(this.pulse);

    const shape = CLASSES[e.cls].shape || 'orb';
    this.model = buildEmitterModel(shape, c3, e.confidence);
    this.model.position.y = 2.6; this.model.scale.setScalar(MODEL_SCALE);
    root.add(this.model);

    this.hit = new THREE.Mesh(new THREE.SphereGeometry(1.7,10,8),
      new THREE.MeshBasicMaterial({ visible:false }));
    this.hit.position.y = 2.6; root.add(this.hit);
    this.orb = this.hit;


    this.ghost = makeWireframeClone(this.model, c3);
    this.ghost.position.y = 2.6; this.ghost.scale.setScalar(MODEL_SCALE);
    this.ghost.visible = false;
    root.add(this.ghost);
    this._occT = 0; this._occRay = new THREE.Raycaster();
    this._wp = new THREE.Vector3(); this._cp = new THREE.Vector3();

    // lock-on shell sized to ENCLOSE the model (radius scales with MODEL_SCALE,
    // otherwise it ends up buried inside the larger airframe)
    this.shell = new THREE.Mesh(new THREE.IcosahedronGeometry(MODEL_SCALE*0.72, 1),
      new THREE.MeshBasicMaterial({ color:c3, wireframe:true, transparent:true,
        opacity:0.0, fog:false, toneMapped:false, blending:THREE.NormalBlending }));
    this.shell.position.y = 2.6; root.add(this.shell);

    this.claimRing = new THREE.Mesh(new THREE.TorusGeometry(MODEL_SCALE*0.80, 0.10, 8, 44),
      new THREE.MeshBasicMaterial({ color:new THREE.Color(CLAIM_COLOR), transparent:false,
        fog:false, toneMapped:false, blending:THREE.NormalBlending }));
    this.claimRing.rotation.x = -Math.PI/2; this.claimRing.position.y = 0.12;
    this.claimRing.visible = false; root.add(this.claimRing);

    this.trailPts = []; this.trailLast = 0;
    const worldParent = this.el.object3D.parent || root;
    this.trailLine = new THREE.Line(new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color:c3, transparent:true, opacity:0.45,
        depthTest:false, fog:false, toneMapped:false }));
    this.trailLine.renderOrder = 6; worldParent.add(this.trailLine);

    this.phase = Math.random()*1800;
    this.focused = false;
  },
  setFocused: function (f) {
    if (this.focused === f) return;
    this.focused = f;
    this.beamMat.uniforms.uActive.value = f ? 1.0 : 0.0;
    this.model.scale.setScalar(f ? MODEL_SCALE*1.22 : MODEL_SCALE);
    this.shell.material.opacity = f ? 0.9 : 0.0;
  },
  toggleClaim: function () {
    this.claimed = !this.claimed;
    this.claimRing.visible = this.claimed;
    const c = new THREE.Color(this.claimed ? CLAIM_COLOR : this.baseColor);
    this.beamMat.uniforms.uColor.value.copy(c);
    this.model.userData.mats.forEach(function(m){
      const ac = c.clone().multiplyScalar(0.88);   // same desaturation as creation-time ACCENT()
      m.color.copy(ac).multiplyScalar(0.12);
      if (m.emissive) m.emissive.copy(ac);         // emissiveIntensity (confidence) stays as-created
    });
    this.ghost.userData.mats.forEach(function(m){ m.color.copy(c); });
    this.spine.material.color.copy(c);
    this.shell.material.color.copy(c);
    this.ring.material.color.copy(c);
    this.pulse.material.color.copy(c);
    return this.claimed;
  },
  currentColor: function () { return this.claimed ? CLAIM_COLOR : this.baseColor; },
  tick: function (time, dt) {
    const S = window.RFX;
    if (!S) return;
    const e = this.e, root = this.el.object3D;

    const active = (e.mode === S.mode);
    if (root.visible !== active) root.visible = active;
    this.trailLine.visible = active && S.showTrails;
    if (!active) return;

    const step = Math.min(dt || 16, 50) / 1000;
    if (e.driven) {
      root.position.set(e.pos[0], 0, e.pos[2]);
    } else if (S.playing) {
      e.pos[0] += e.vel[0] * step * S.speed;
      e.pos[2] += e.vel[2] * step * S.speed;
      e.vel[0] += Math.sin(time*0.0005 + this.phase)*0.02*step;
      e.vel[2] += Math.cos(time*0.0004 + this.phase)*0.02*step;
      if (Math.hypot(e.pos[0], e.pos[2]) > 75) { e.vel[0]*=-1; e.vel[2]*=-1; }
      root.position.set(e.pos[0], 0, e.pos[2]);
    }
    const ts = time * 0.001;
    this.beamMat.uniforms.uTime.value = ts;
    this.beam.visible = S.beams;
    this.spine.visible = S.beams;
    const t = ((time + this.phase) % 1800) / 1800;
    const s = 1 + t*4.0;
    this.pulse.scale.set(s,1,s);
    this.pulse.material.opacity = 0.85 * (1 - t);
    const bob = e.driven
      ? Math.max(2.0, e.pos[1])
      : 2.6 + Math.sin(time*0.0015 + this.phase)*0.10;
    this.model.position.y = this.hit.position.y = bob;
    this.shell.position.y = this.ghost.position.y = bob;
    if (e.driven) {
      const k = bob / 30;
      this.beam.scale.y = k;  this.beam.position.y = bob/2;
      this.spine.scale.y = k; this.spine.position.y = bob/2;
    }
    this.model.rotation.y += step * 0.4;
    this.ghost.rotation.y = this.model.rotation.y;

    if (time - this._occT > 250) {
      this._occT = time;
      let occluded = false;
      const occ = this.el.sceneEl.components['real-occluders'];
      const cam2 = this.el.sceneEl.camera;
      if (occ && occ.entries.size && cam2) {
        cam2.getWorldPosition(this._cp);
        this.model.getWorldPosition(this._wp);
        const dir = this._wp.clone().sub(this._cp);
        const dist = dir.length();
        this._occRay.set(this._cp, dir.normalize());
        this._occRay.far = dist - 0.05;
        occluded = this._occRay.intersectObjects(occ.group.children, false).length > 0;
      }
      this.ghost.visible = occluded;
    }
    this.shell.rotation.y += step * (this.focused ? 1.1 : 0.35);
    this.shell.rotation.x += step * 0.15;
    if (this.claimed) {
      this.claimRing.rotation.z += step * 0.8;
      this.claimRing.scale.setScalar(1 + Math.sin(time*0.004)*0.05);
    }

    if (time - this.trailLast > 150) {
      this.trailLast = time;
      this.trailPts.push(new THREE.Vector3(root.position.x, bob, root.position.z));
      if (this.trailPts.length > 40) this.trailPts.shift();
      // Fresh BufferGeometry each update, not setFromPoints() on the old one:
      // reusing a geometry whose point count keeps growing hits a fixed
      // buffer capacity and silently truncates (BufferGeometry warns
      // "buffer size too small" and drops the extra points). Trivial cost
      // at 6 drones / <=121 vertices each.
      let newGeo;
      if (this.trailPts.length > 2) {
        // Catmull-Rom through the raw samples so the trail reads as one
        // continuous curve rather than a faceted polyline.
        const curve = new THREE.CatmullRomCurve3(this.trailPts, false, 'catmullrom', 0.5);
        newGeo = new THREE.BufferGeometry().setFromPoints(curve.getPoints(this.trailPts.length * 3));
      } else if (this.trailPts.length > 1) {
        newGeo = new THREE.BufferGeometry().setFromPoints(this.trailPts);
      }
      if (newGeo) {
        this.trailLine.geometry.dispose();
        this.trailLine.geometry = newGeo;
      }
    }
  }
});
