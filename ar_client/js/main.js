/* main.js — flat/desktop page controls (MODE, FORMATION), reachable
   before ever entering XR. Loaded at the end of body, after #overlay. */
(function () {
  function wire() {
    if (!window.RFX) { setTimeout(wire, 200); return; }
    const modeBtn = document.getElementById('btn-mode');
    const formBtn = document.getElementById('btn-formation');
    modeBtn.addEventListener('click', function () {
      window.RFX.mode = window.RFX.mode === 'swarm' ? 'blobs' : 'swarm';
    });
    formBtn.addEventListener('click', function () {
      const sc = document.querySelector('a-scene');
      const sp = sc && sc.components['swarm-player'];
      if (!sp || typeof FORMATION_NAMES === 'undefined') return;
      const cur = sp.pendingFormation || sp.formation;
      const idx = FORMATION_NAMES.indexOf(cur);
      sp.selectFormation(FORMATION_NAMES[(idx+1) % FORMATION_NAMES.length]);
      window.RFX.mode = 'swarm';
    });
    setInterval(function () {
      const S = window.RFX;
      modeBtn.textContent = 'MODE: ' + (S.mode==='swarm' ? 'SWARM FORMATION' : 'LIVE LOCALISATION');
      const sc = document.querySelector('a-scene');
      const sp = sc && sc.components['swarm-player'];
      const name = sp ? (sp.pendingFormation || sp.formation) : 'v_shape';
      formBtn.textContent = 'FORMATION: ' + name.toUpperCase().replace('_',' ') + (S.transState==='transitioning' ? ' \u2026' : '');
    }, 250);
  }
  wire();
})();
