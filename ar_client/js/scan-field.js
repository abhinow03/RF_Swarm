/* scan-field.js — the ground scan-ring shader effect, and its sole-consumer material helper. */
function makeScanMaterial() {
  return new THREE.ShaderMaterial({
    transparent:true, depthWrite:false, blending:THREE.AdditiveBlending, side:THREE.DoubleSide,
    uniforms: { uColor:{ value:new THREE.Color('#2dffa0') }, uTime:{ value:0 } },
    vertexShader: [
      'varying vec2 vUv;',
      'void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }'
    ].join('\n'),
    fragmentShader: [
      'uniform vec3 uColor; uniform float uTime;',
      'varying vec2 vUv;',
      'void main(){',
      '  vec2 p = vUv - 0.5;',
      '  float r = length(p) * 2.0;',
      '  if (r > 1.0) discard;',
      '  float ring = sin(r * 26.0 - uTime * 1.6);',
      '  ring = smoothstep(0.86, 1.0, ring) * 0.5;',
      '  vec2 g = abs(fract(vUv * 26.0) - 0.5);',
      '  float grid = smoothstep(0.46, 0.5, max(g.x, g.y)) * 0.10;',
      '  float ang = atan(p.y, p.x);',
      '  float sweep = smoothstep(0.90, 1.0, cos(ang - uTime * 0.7));',
      '  sweep *= (1.0 - r) * 0.35;',
      '  float falloff = pow(1.0 - r, 1.4);',
      '  float a = (ring + grid + sweep) * falloff * 0.55;',
      '  gl_FragColor = vec4(uColor, a);',
      '}'
    ].join('\n')
  });
}

AFRAME.registerComponent('scan-field', {
  init: function () {
    this.mat = makeScanMaterial();
    const m = new THREE.Mesh(new THREE.PlaneGeometry(160,160), this.mat);
    m.rotation.x = -Math.PI/2; m.position.y = 0.01;
    this.el.setObject3D('scan', m);
  },
  tick: function (time) { this.mat.uniforms.uTime.value = time * 0.001; }
});
