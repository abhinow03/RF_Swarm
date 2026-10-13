/* formation-banner.js — compact classifier readout, now docked top-right
   instead of floating centre (see markup in index.html for position). */
AFRAME.registerComponent('formation-banner', {
  init: function () {
    this.panel = makeCanvasPanel(0.38, 0.155, 2200);
    this.panel.mesh.visible = false;
    this.el.object3D.add(this.panel.mesh);
    this.last = 0;
  },
  tick: function (time) {
    const S = window.RFX;
    const show = !!(S && S.transState && S.mode === 'swarm');
    this.panel.mesh.visible = show;
    if (!show || time - this.last < 120) return;
    this.last = time;
    this.draw(S);
  },
  draw: function (S) {
    const ctx=this.panel.ctx, cv=this.panel.cv, W=cv.width, H=cv.height;
    ctx.clearRect(0,0,W,H);
    ctx.fillStyle='rgba(2,10,7,0.93)';
    roundRect(ctx,3,3,W-6,H-6,16); ctx.fill();

    if (S.transState === 'transitioning') {
      ctx.strokeStyle='#ffb454'; ctx.lineWidth=4; ctx.stroke();
      ctx.textBaseline='middle';
      ctx.fillStyle='#ffb454'; ctx.textAlign='left';
      ctx.font='bold ' + Math.round(H*0.16) + 'px ui-monospace, monospace';
      ctx.fillText('TRANSITIONING', W*0.035, H*0.15);
      ctx.textAlign='right';
      ctx.fillText(Math.round(S.transAlpha*100) + '%', W*0.965, H*0.15);
      ctx.fillStyle='#eafff6'; ctx.textAlign='left';
      ctx.font = Math.round(H*0.09) + 'px ui-monospace, monospace';
      ctx.fillText(S.transFrom.toUpperCase().replace('_',' ') + '  \u2192  ' + S.transTo.toUpperCase().replace('_',' '),
        W*0.035, H*0.30);
      const bx=W*0.035, bw=W*0.93, by=H*0.42, bh=H*0.09;
      ctx.fillStyle='rgba(255,255,255,0.10)'; roundRect(ctx,bx,by,bw,bh,bh/2); ctx.fill();
      ctx.fillStyle='#ffb454'; roundRect(ctx,bx,by,Math.max(bh,bw*S.transAlpha),bh,bh/2); ctx.fill();
      ctx.textAlign='center'; ctx.fillStyle='#7fae9c';
      ctx.font = Math.round(H*0.05) + 'px ui-monospace, monospace';
      ctx.fillText('live cosine blend \u2014 not classifier output', W/2, H*0.62);
      this.panel.tex.needsUpdate = true;
      return;
    }

    ctx.strokeStyle='#39e0ff'; ctx.lineWidth=4; ctx.stroke();
    ctx.textBaseline='middle';
    const top = (S.classConf && S.classLabel && S.classConf.find(c=>c[0]===S.classLabel)) || [S.classLabel,0];
    ctx.fillStyle='#39e0ff'; ctx.textAlign='left';
    ctx.font='bold ' + Math.round(H*0.155) + 'px ui-monospace, monospace';
    ctx.fillText('CLASSIFIED  ' + (S.classLabel||'\u2014').toUpperCase().replace('_',' '), W*0.035, H*0.13);
    ctx.textAlign='right';
    ctx.fillText(Math.round(top[1]*100) + '%', W*0.965, H*0.13);

    if (S.classConf) {
      const rows = S.classConf.slice().sort((a,b)=> a[0]<b[0]?-1:1);
      const rowTop = H*0.28, rowH = (H*0.68)/rows.length;
      rows.forEach((r,i) => {
        const y = rowTop + i*rowH;
        const isTop = r[0] === S.classLabel;
        ctx.textAlign='left'; ctx.fillStyle = isTop ? '#eafff6' : '#7fae9c';
        ctx.font = (isTop?'bold ':'') + Math.round(H*0.052) + 'px ui-monospace, monospace';
        ctx.fillText(r[0].toUpperCase().replace('_',' '), W*0.035, y+rowH*0.5);
        const bx = W*0.46, bw = W*0.44, bh = rowH*0.5;
        ctx.fillStyle = 'rgba(255,255,255,0.08)';
        roundRect(ctx, bx, y+rowH*0.25, bw, bh, bh/2); ctx.fill();
        ctx.fillStyle = isTop ? '#39e0ff' : '#3a5b50';
        roundRect(ctx, bx, y+rowH*0.25, Math.max(bh, bw*r[1]), bh, bh/2); ctx.fill();
        ctx.textAlign='right'; ctx.fillStyle = isTop ? '#39e0ff' : '#7fae9c';
        ctx.font = Math.round(H*0.045) + 'px ui-monospace, monospace';
        ctx.fillText(Math.round(r[1]*100)+'%', W*0.965, y+rowH*0.5);
      });
    }
    this.panel.tex.needsUpdate = true;
  }
});
