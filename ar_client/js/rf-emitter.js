/* rf-emitter.js — the per-emitter beacon: model, beam, occlusion ghost,
   claim state, breadcrumb trail. Includes its sole-consumer helpers
   (model builder, wireframe clone, beam shader) rather than utils.js. */

// Confidence LED gradient for live RF_Swarm emitters — reuses feed-status.js's
// established "live/good" green rather than inventing a new color; low end is a
// coral distinct from CLAIM_COLOR's amber so a low-confidence emitter is never
// visually confused with a claimed one.
const LIVE_GOOD_COLOR = '#2dffa0';
const LIVE_BAD_COLOR = '#ff6b5c';
function lerpColorHex(c1, c2, t) {
  const a = new THREE.Color(c1), b = new THREE.Color(c2);
  return a.lerp(b, Math.max(0, Math.min(1, t)));
}
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

function buildEmitterModel(shape, color) {
  const g = new THREE.Group(); const mats = [];
  // Flat-shaded LIT material, not MeshBasic: with flatShading every facet
  // catches the directional light differently, so the model reads as a solid
  // 3D volume instead of a flat single-colour sticker. Strong emissive floor
  // guarantees it never goes dark even facing away from the light — the look
  // is opaque + saturated + facet-varied, which is what survives passthrough.
  const M = (o) => {
    const m = new THREE.MeshLambertMaterial({ color:color.clone(),
      emissive:color.clone().multiplyScalar(0.55), flatShading:true,
      fog:false, toneMapped:false, transparent:o<1, opacity:o });
    mats.push(m); return m;
  };
  const WHITE = () => new THREE.MeshLambertMaterial({ color:0xffffff,
    emissive:0x888888, flatShading:true, fog:false, toneMapped:false });

  if (shape === 'drone') {
    g.add(new THREE.Mesh(new THREE.BoxGeometry(0.5,0.16,0.5), M(1)));
    const armG = new THREE.CylinderGeometry(0.035,0.035,0.9,6);
    for (let i=0;i<2;i++){
      const arm = new THREE.Mesh(armG, M(1));
      arm.rotation.z = Math.PI/2; arm.rotation.y = (i===0?1:-1)*Math.PI/4;
      g.add(arm);
    }
    const rotG = new THREE.TorusGeometry(0.22,0.028,6,14);
    [[0.45,0.45],[0.45,-0.45],[-0.45,0.45],[-0.45,-0.45]].forEach(function(p){
      const r = new THREE.Mesh(rotG, M(1));
      r.rotation.x = Math.PI/2; r.position.set(p[0],0.10,p[1]);
      g.add(r);
      const hub = new THREE.Mesh(new THREE.SphereGeometry(0.05,6,5), WHITE());
      hub.position.set(p[0],0.10,p[1]); g.add(hub);
    });
  } else if (shape === 'remote') {
    g.add(new THREE.Mesh(new THREE.BoxGeometry(0.5,0.14,0.75), M(1)));
    const antG = new THREE.CylinderGeometry(0.025,0.025,0.55,6);
    [-0.16,0.16].forEach(function(x,i){
      const a = new THREE.Mesh(antG, M(1));
      a.position.set(x,0.32,-0.28); a.rotation.x = -0.5; a.rotation.z = i===0?0.25:-0.25;
      g.add(a);
      const tip = new THREE.Mesh(new THREE.SphereGeometry(0.045,6,5), WHITE());
      tip.position.set(x + (i===0?0.13:-0.13), 0.55, -0.53); g.add(tip);
    });
    [-0.13,0.13].forEach(function(x){
      const stick = new THREE.Mesh(new THREE.SphereGeometry(0.06,8,6), WHITE());
      stick.position.set(x,0.11,0.16); g.add(stick);
    });
  } else if (shape === 'antenna') {
    const base = new THREE.Mesh(new THREE.BoxGeometry(0.4,0.14,0.4), M(1));
    base.position.y = -0.35; g.add(base);
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.04,0.05,0.8,6), M(1));
    pole.position.y = 0.05; g.add(pole);
    const tip = new THREE.Mesh(new THREE.SphereGeometry(0.07,8,6), WHITE());
    tip.position.y = 0.48; g.add(tip);
    [0.22,0.38,0.54].forEach(function(r){
      const arc = new THREE.Mesh(new THREE.TorusGeometry(r,0.022,6,14,Math.PI*0.55), M(0.95));
      arc.position.y = 0.48; arc.rotation.z = Math.PI*0.225;
      g.add(arc);
    });
  } else {
    g.add(new THREE.Mesh(new THREE.SphereGeometry(0.55,20,16), M(1)));
    const eq = new THREE.Mesh(new THREE.TorusGeometry(0.72,0.03,6,24), M(0.95));
    eq.rotation.x = Math.PI/2; g.add(eq);
  }
  g.userData.mats = mats;
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

    // uncertainty ring (live RF_Swarm emitters only): a thin outline whose RADIUS
    // scales with pos_sigma, so a wide ring reads at a glance as "the system isn't
    // sure where this is" — no new color, one geometric encoding. Starts invisible;
    // tick() sizes/shows it only for driven emitters carrying a real posSigma.
    this.uncRing = new THREE.Mesh(new THREE.RingGeometry(2.35,2.5,48),
      new THREE.MeshBasicMaterial({ color:LIVE_GOOD_COLOR, transparent:true, opacity:0.0,
        side:THREE.DoubleSide, depthTest:false, fog:false, toneMapped:false }));
    this.uncRing.renderOrder = 8;
    this.uncRing.rotation.x = -Math.PI/2; this.uncRing.position.y = 0.04; root.add(this.uncRing);

    const shape = CLASSES[e.cls].shape || 'orb';
    this.model = buildEmitterModel(shape, c3);
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
      m.color.copy(c);
      if (m.emissive) m.emissive.copy(c).multiplyScalar(0.55);
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

    // live-only encoding: track STATUS -> form/opacity, CONFIDENCE -> LED color,
    // pos_sigma -> uncertainty ring radius. Two encodings, no new colors beyond
    // the existing LIVE_GOOD/LIVE_BAD pair. No-op for scripted/blobs emitters,
    // which never set e.status.
    this._statusLost = false;
    if (e.driven && e.status !== undefined) {
      const coasting = e.status === 2, lost = e.status === 3;
      this._statusLost = lost;
      const modelOpacity = coasting ? 0.35 : 1.0;
      this.model.userData.mats.forEach(function (m) {
        m.transparent = modelOpacity < 1; m.opacity = modelOpacity;
      });
      this.model.visible = !lost;
      this.pulse.visible = !coasting && !lost;          // "no LED" while coasting/lost
      this.ring.visible = !lost;
      if (!this.claimed) {
        const conf = e.confidence != null ? e.confidence : 1.0;
        this.ring.material.color.copy(lerpColorHex(LIVE_BAD_COLOR, LIVE_GOOD_COLOR, conf));
      }
      if (e.posSigma != null && !lost) {
        // sigma in metres -> ring radius scale; clamped so it stays legible at
        // both ends (a near-zero sigma still shows a thin ring, a huge one
        // doesn't blow past the model's own footprint)
        const sc = Math.max(1.0, Math.min(2.2, 1.0 + e.posSigma / 25));
        this.uncRing.scale.set(sc, 1, sc);
        this.uncRing.material.opacity = 0.4;
      } else {
        this.uncRing.material.opacity = 0.0;
      }
    }

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
      this.ghost.visible = occluded || this._statusLost;
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
