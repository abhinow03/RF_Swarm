/* Headless end-to-end test of rf_bridge.js against the LIVE ws rosbridge.
 * Exercises the EXACT browser code path: RFBridge.connect() -> ROSLIB subscribe ->
 * the parsers -> the onScene/onFormation/onNarration callbacks. Only the DOM render
 * differs from test.html. Run while the pipeline + a ws rosbridge (:9090) are up.
 *   node capstone_ws/client/headless_test.mjs
 */
import * as ROSLIBmod from 'roslib';
globalThis.ROSLIB = ROSLIBmod.default || ROSLIBmod;
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const RFBridge = require('./rf_bridge.js');

const seen = { scene: 0, form: 0, narr: 0 };
const firstAt = {};
const t0 = Date.now();
const stamp = () => ((Date.now() - t0) / 1000).toFixed(1) + 's';

const bridge = RFBridge.connect({
  url: 'ws://localhost:9090',
  onStatus: (s) => console.log(`[${stamp()}] status: ${s}`),
  onScene: (env, emitters) => {
    seen.scene++;
    if (seen.scene === 1) { firstAt.scene = stamp();
      console.log(`[${stamp()}] /scene_update  scenario=${env.scenario_id} k=${env.k_estimated} ntracks=${emitters.length}`);
      console.log(`             track0=${emitters[0].id} cls=${emitters[0].cls} posYup=[${emitters[0].pos.map(v=>v.toFixed(1))}] status=${emitters[0].status} gdop=${emitters[0].gdop.toFixed(2)}`);
    }
  },
  onFormation: (f) => {
    seen.form++;
    if (seen.form === 1) { firstAt.form = stamp();
      console.log(`[${stamp()}] /formation     ${f.formation_class} conf=${f.confidence.toFixed(3)} nprobs=${f.class_probabilities.length} from_truth=${f.from_truth}`);
    }
  },
  onNarration: (n) => {
    seen.narr++;
    if (seen.narr === 1) { firstAt.narr = stamp();
      console.log(`[${stamp()}] /narration     [${n.template_id}] ${(n.text||'').slice(0,60)}`);
    }
  },
});

setTimeout(() => {
  console.log(`\ncounts: scene=${seen.scene} formation=${seen.form} narration=${seen.narr}`);
  console.log(`first-arrival: scene=${firstAt.scene||'-'} formation=${firstAt.form||'-'} narration=${firstAt.narr||'-'}`);
  const independent = seen.scene > seen.form && seen.scene > seen.narr;
  console.log(`three separate subscriptions received: ${['scene','form','narr'].every(k=>seen[k]>0)}`);
  console.log(`independent rates (scene ticks >> formation/narration, per Option C): ${independent}`);
  bridge.close();
  process.exit(0);
}, 55000);
