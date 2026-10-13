/* emitter-source.js — spawns all emitter entities and owns the shared window.RFX state object. */
AFRAME.registerComponent('emitter-source', {
  init: function () {
    const self = this;
    window.RFX = {
      playing:true, speed:1, beams:true, isAR:false, anchored:false,
      gazed:null, claimProgress:0, scale:CONFIG.scale, showStatus:CONFIG.showStatus, arEnterTime:0,
      mode:CONFIG.mode,
      transState:null, transFrom:null, transTo:null, transAlpha:0,
      selectedFormation:'v_shape',
      liveFeed:false,   // set true by ros-feed.js once the rosbridge connection is up
      classLabel:null, classConf:null,
      showOverlay:CONFIG.showOverlay, showTrails:CONFIG.showTrails,
      home: ALL_EMITTERS.map(function (e) { return { pos:e.pos.slice(), vel:e.vel.slice() }; })
    };
    ALL_EMITTERS.forEach(function (e,i) {
      const el = document.createElement('a-entity');
      el.setAttribute('position', e.pos[0] + ' 0 ' + e.pos[2]);
      el.setAttribute('rf-emitter', 'idx: ' + i);
      self.el.appendChild(el);
    });
    setTimeout(function(){ self.el.sceneEl.emit('targets-ready'); }, 400);
  }
});
