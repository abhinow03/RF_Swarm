/* rf_bridge.js — RF_Swarm live-data client for the AR/WebXR viz.
 *
 * DROP-IN for AR-visuliazation (abhinow03). Connects to the capstone_bridge over
 * rosbridge (wss:// for the Quest; ws:// works for a local desktop test) and feeds
 * the THREE client-facing topics to your scene via callbacks. Nothing here does
 * frame maths — /scene_update track pos/vel arrive ALREADY in Three.js Y-up
 * ([E,U,-N]) from the server (Option C). formation/narration are their OWN latched
 * subscriptions, so they update independently of the 10 Hz tracks tick.
 *
 * Usage (browser, after loading roslib.min.js and this file):
 *   const bridge = RFBridge.connect({
 *     url: 'wss://mars-4090:9443',            // or ws://localhost:9090 for a local test
 *     onStatus:    (s) => {...},              // 'connecting'|'connected'|'closed'|'error'
 *     onScene:     (env, emitters) => {...},  // emitters already in your {id,cls,pos,vel,...} shape
 *     onFormation: (f) => {...},              // {formation_class, confidence, class_probabilities, member_ids, from_truth}
 *     onNarration: (n) => {...},              // {text, template_id, source, severity}
 *   });
 *   // bridge.close() to disconnect.
 *
 * The pure parsers (trackToEmitter / parseScene / parseFormation / parseNarration)
 * are exported for node unit-testing and carry NO roslib dependency.
 */
(function (root) {
  'use strict';

  // Reserved fallback class for malformed payloads (kept distinct from the real
  // 'swarm' look, per the wire-format contract). If the host page defines CLASSES,
  // honor it; otherwise accept the bridge's constant 'swarm'.
  function classOf(cls) {
    var known = (typeof CLASSES !== 'undefined') ? !!CLASSES[cls] : (cls === 'swarm');
    return known ? cls : 'unknown';
  }

  // wire-track -> the AR client's emitter object contract. pos/vel are already Y-up.
  function trackToEmitter(t) {
    return {
      id: t.id,
      cls: classOf(t.cls),
      confidence: (typeof t.confidence === 'number') ? t.confidence : 0,
      pos: Array.isArray(t.pos) ? t.pos.slice() : [0, 0, 0],
      vel: Array.isArray(t.vel) ? t.vel.slice() : [0, 0, 0],
      driven: true,
      mode: 'swarm',
      // render-only extras carried with safe defaults
      status: t.status || 'tracking',
      pos_sigma: (typeof t.pos_sigma === 'number') ? t.pos_sigma : 0,
      gdop: (typeof t.gdop === 'number') ? t.gdop : 0,
    };
  }

  // /scene_update is std_msgs/String whose .data is the Option-C JSON envelope.
  function parseScene(msg) {
    var env = (typeof msg.data === 'string') ? JSON.parse(msg.data) : msg;
    var emitters = (env.tracks || []).map(trackToEmitter);
    return { env: env, emitters: emitters };
  }

  function parseFormation(msg) {
    return {
      formation_class: msg.formation_class,
      confidence: msg.confidence,
      class_probabilities: msg.class_probabilities || [],
      member_ids: msg.member_ids || [],
      from_truth: !!msg.from_truth,
    };
  }

  function parseNarration(msg) {
    return {
      text: msg.text,
      template_id: msg.template_id,
      source: msg.source,
      severity: msg.severity,
    };
  }

  // --- browser connection layer (needs ROSLIB) ---
  function connect(opts) {
    if (typeof ROSLIB === 'undefined') {
      throw new Error('rf_bridge: ROSLIB not loaded — include roslib.min.js before rf_bridge.js');
    }
    var url = opts.url || 'wss://localhost:9443';
    var status = opts.onStatus || function () {};
    status('connecting');

    var ros = new ROSLIB.Ros({ url: url });
    ros.on('connection', function () { status('connected'); });
    ros.on('close', function () { status('closed'); });
    ros.on('error', function (e) { status('error', e); });

    // THREE independent subscriptions (separate topics => independent update rates).
    var scene = new ROSLIB.Topic({ ros: ros, name: '/scene_update', messageType: 'std_msgs/String' });
    var form = new ROSLIB.Topic({ ros: ros, name: '/formation', messageType: 'capstone_msgs/FormationEstimate' });
    var narr = new ROSLIB.Topic({ ros: ros, name: '/narration', messageType: 'capstone_msgs/Narration' });

    if (opts.onScene) scene.subscribe(function (m) { var p = parseScene(m); opts.onScene(p.env, p.emitters); });
    if (opts.onFormation) form.subscribe(function (m) { opts.onFormation(parseFormation(m)); });
    if (opts.onNarration) narr.subscribe(function (m) { opts.onNarration(parseNarration(m)); });

    return {
      ros: ros,
      close: function () { try { scene.unsubscribe(); form.unsubscribe(); narr.unsubscribe(); ros.close(); } catch (e) {} },
    };
  }

  var api = {
    connect: connect,
    trackToEmitter: trackToEmitter,
    parseScene: parseScene,
    parseFormation: parseFormation,
    parseNarration: parseNarration,
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = api;  // node
  else root.RFBridge = api;                                                    // browser
})(typeof window !== 'undefined' ? window : this);
