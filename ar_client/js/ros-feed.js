/* ros-feed.js — live RF_Swarm feed over rosbridge (wss). When connected, this
   OWNS the 6 'swarm' emitters' pos/vel (they're already `driven:true`, the
   same seam swarm-player.js normally writes through) and REPLACES the
   client-side scripted animation entirely — window.RFX.liveFeed gates
   swarm-player.js off (see the guard added at the top of its tick()).

   Wire contract (frozen, RF_Swarm capstone_bridge):
     /scene_update  std_msgs/msg/String, JSON:
       { tracks: [ { id:"d1", cls:"swarm", confidence, pos:[x,y,z],
                     vel:[vx,vy,vz], status, pos_sigma, gdop }, ... ] }
       pos/vel are ALREADY server-side ENU->Y-up converted — used as-is,
       no rosEnuToScene() conversion here (that would double-convert).
     /formation     capstone_msgs FormationEstimate (type auto-detected):
       { formation_class, confidence, class_probabilities[8], from_truth }
       class index order == FORMATION_NAMES[0..6] + "transitioning"[7]
       (config.js's FORMATION_NAMES is the first 7 of these, same order).

   On disconnect, liveFeed drops to false and swarm-player.js's own scripted
   demo resumes untouched — this file never mutates SWARM_EMITTERS' shape,
   only pos/vel/classification while connected. */
(function () {
  function idIndex(trackId) {
    // "d1".."d6" -> 0..5, matching SWARM_EMITTERS' fixed 1..6 build order.
    const m = /(\d+)\s*$/.exec(trackId || '');
    return m ? (parseInt(m[1], 10) - 1) : -1;
  }

  function applyScene(msg) {
    const S = window.RFX;
    if (!S) return;
    let data;
    try { data = JSON.parse(msg.data); } catch (err) { return; }
    const tracks = data.tracks || [];
    tracks.forEach(function (t) {
      const i = idIndex(t.id);
      if (i < 0 || i >= SWARM_EMITTERS.length) return;
      const e = SWARM_EMITTERS[i];
      if (Array.isArray(t.pos) && t.pos.length === 3) e.pos[0] = t.pos[0], e.pos[1] = t.pos[1], e.pos[2] = t.pos[2];
      if (Array.isArray(t.vel) && t.vel.length === 3) e.vel[0] = t.vel[0], e.vel[1] = t.vel[1], e.vel[2] = t.vel[2];
      e.confidence = t.confidence;
      e.status = t.status; e.posSigma = t.pos_sigma; e.gdop = t.gdop;
    });
    S.transState = 'settled';
  }

  // The real classifier reads noisy EKF-tracked positions (not idealized
  // formation shapes), so raw per-message labels can flicker between two
  // adjacent classes (documented Phase-5 finding: e.g. encirclement <->
  // dispersed under EKF position RMSE). Committing on a majority vote over
  // a short rolling window turns that flicker into a stable hold, without
  // faking a transition that isn't actually happening. class_probabilities
  // (the actual confidence distribution) is applied on every message
  // unthrottled — only the single displayed classLabel is debounced.
  const FORMATION_HISTORY_LEN = 5;
  const FORMATION_VOTE_MIN = 3;   // >=3 of last 5 must agree to switch
  let formationHistory = [];
  let committedLabel = null;

  function applyFormation(msg) {
    const S = window.RFX;
    if (!S || typeof FORMATION_NAMES === 'undefined') return;
    if (Array.isArray(msg.class_probabilities)) {
      S.classConf = FORMATION_NAMES.map(function (name, i) {
        return [name, msg.class_probabilities[i] || 0];
      });
    }
    formationHistory.push(msg.formation_class);
    if (formationHistory.length > FORMATION_HISTORY_LEN) formationHistory.shift();
    if (committedLabel === null) { committedLabel = msg.formation_class; }
    else if (msg.formation_class !== committedLabel) {
      const agree = formationHistory.filter(function (l) { return l === msg.formation_class; }).length;
      if (agree >= FORMATION_VOTE_MIN) committedLabel = msg.formation_class;
    }
    S.classLabel = committedLabel;
  }

  function hideScriptedOverlay() {
    const sc = document.querySelector('a-scene');
    const sp = sc && sc.components && sc.components['swarm-player'];
    if (sp && sp.overlay) { sp.overlay.visible = false; sp.centroidDot.visible = false; }
  }

  function connect() {
    if (!window.RFX) { setTimeout(connect, 200); return; }
    const url = CONFIG.rosbridgeUrl;
    if (!url) return;
    let ws;
    try { ws = new WebSocket(url); } catch (err) { scheduleReconnect(); return; }

    ws.onopen = function () {
      console.log('[ros-feed] connected:', url);
      ws.send(JSON.stringify({ op: 'subscribe', topic: CONFIG.rosbridgeTopic, type: CONFIG.rosbridgeMsgType }));
      ws.send(JSON.stringify({ op: 'subscribe', topic: '/formation' }));   // type auto-detected
      formationHistory = []; committedLabel = null;   // fresh vote state per connection
      window.RFX.liveFeed = true;
      hideScriptedOverlay();
    };
    ws.onmessage = function (evt) {
      let env;
      try { env = JSON.parse(evt.data); } catch (err) { return; }
      if (env.op !== 'publish') return;
      if (env.topic === CONFIG.rosbridgeTopic) applyScene(env.msg);
      else if (env.topic === '/formation') applyFormation(env.msg);
    };
    ws.onclose = function () {
      console.warn('[ros-feed] disconnected — falling back to scripted demo');
      window.RFX.liveFeed = false;
      scheduleReconnect();
    };
    ws.onerror = function () { ws.close(); };
  }

  let reconnectTimer = null;
  function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectTimer = setTimeout(function () { reconnectTimer = null; connect(); }, 3000);
  }

  connect();
})();
