/* world-scale.js — applies the live scale factor to the world root. */
AFRAME.registerComponent('world-scale', {
  tick: function () {
    const S = window.RFX;
    if (!S) return;
    const s = S.scale;
    const o = this.el.object3D;
    if (o.scale.x !== s) o.scale.set(s, s, s);
  }
});
