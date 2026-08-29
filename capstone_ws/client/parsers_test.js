/* Node unit test for rf_bridge.js pure parsers (no browser/roslib needed).
 * Run: node capstone_ws/client/parsers_test.js  */
const B = require('./rf_bridge.js');
let fail = 0;
function ok(cond, msg) { console.log((cond ? '  OK  ' : ' FAIL ') + msg); if (!cond) fail++; }

// real /scene_update envelope shape (from the live pipeline)
const env = {
  schema_version: 1, scenario_id: 'gate_k6', k_estimated: 6, pipeline_latency_ms: 100.0,
  tracks: [
    { id: 'd1', cls: 'swarm', confidence: 0.963, pos: [20.97, 59.67, 12.72], vel: [1.0, 0.0, 0.0],
      status: 'tracking', pos_sigma: 2.4, gdop: 1.33 },
    { id: 'd2', cls: 'swarm', confidence: 0.0, pos: [-9.4, 32.0, 1.1], vel: [1, 0, 0], status: 'init' },
  ],
};
const sceneMsg = { data: JSON.stringify(env) };

const p = B.parseScene(sceneMsg);
ok(p.emitters.length === 2, 'parseScene maps all tracks');
const e0 = p.emitters[0];
ok(e0.id === 'd1', 'id carried');
ok(e0.cls === 'swarm', "cls 'swarm' accepted (known)");
ok(JSON.stringify(e0.pos) === JSON.stringify([20.97, 59.67, 12.72]), 'Y-up pos preserved (no client frame math)');
ok(e0.driven === true && e0.mode === 'swarm', 'driven/mode set per contract');
ok(e0.confidence === 0.963, 'confidence carried');
ok(e0.gdop === 1.33 && e0.pos_sigma === 2.4, 'gdop/pos_sigma carried');
ok(p.emitters[1].status === 'init' && p.emitters[1].pos_sigma === 0, 'defaults applied when fields absent');

// malformed cls -> reserved 'unknown' fallback
ok(B.trackToEmitter({ id: 'x', cls: 'garbage', pos: [0, 0, 0], vel: [0, 0, 0] }).cls === 'unknown',
   "malformed cls -> 'unknown' fallback");
ok(B.trackToEmitter({ id: 'x' }).pos.length === 3, 'missing pos -> [0,0,0] default');

// formation / narration
const f = B.parseFormation({ formation_class: 'encirclement', confidence: 0.84,
  class_probabilities: [0.1, 0.84, 0, 0, 0, 0, 0, 0], member_ids: ['d1', 'd2'], from_truth: false });
ok(f.formation_class === 'encirclement' && f.class_probabilities.length === 8, 'parseFormation ok (8 probs)');
const n = B.parseNarration({ text: 'hold', template_id: 'ctx_formation_stgt', source: 'template', severity: 0 });
ok(n.template_id === 'ctx_formation_stgt', 'parseNarration ok');

console.log(fail === 0 ? '\nALL PARSER TESTS PASS' : `\n${fail} FAILURES`);
process.exit(fail === 0 ? 0 : 1);
