/* feed-status.js — small always-on corner tag: LIVE (real RF_Swarm data via
   ros-feed.js) vs SIM (scripted demo). The two look similar enough in motion
   that this is the only reliable way to tell them apart at a glance. */
AFRAME.registerComponent('feed-status', {
  init: function () {
    this.panel = makeCanvasPanel(0.20, 0.055, 2200);
    this.el.object3D.add(this.panel.mesh);
    this.last = -1;
  },
  tick: function () {
    const S = window.RFX;
    if (!S) return;
    const live = !!S.liveFeed;
    if (live === this.last) return;
    this.last = live;
    const ctx = this.panel.ctx, cv = this.panel.cv, W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = 'rgba(2,10,7,0.85)';
    roundRect(ctx, 2, 2, W - 4, H - 4, 14); ctx.fill();
    const color = live ? '#2dffa0' : '#7fae9c';
    ctx.strokeStyle = color; ctx.lineWidth = 4; ctx.stroke();
    ctx.fillStyle = color; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.font = 'bold ' + Math.round(H * 0.42) + 'px ui-monospace, monospace';
    ctx.fillText(live ? '● LIVE RF_SWARM FEED' : '○ SCRIPTED DEMO', W / 2, H / 2);
    this.panel.tex.needsUpdate = true;
  }
});
