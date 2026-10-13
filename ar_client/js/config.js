/* config.js — constants and shared demo data.
   Pure data: no functions, no DOM/THREE calls, safe to load first. */
const CONFIG = {
  bg:'#050807', fogDensity:0.010,
  gazeFar:250, dwellMs:280, claimDwellMs:1400, uiDwellMs:650,
  cursorSpeed: 0.75,       // metres/sec of panel travel at full stick deflection
  scale: 0.05, scaleMin: 0.008, scaleMax: 1.0,
  classifyEveryMs: 200,
  blendSeconds: 4.0,       // fly-to duration, matches the cosine S-curve
  sweepLeadSeconds: 0.7,   // horizontal pre-movement before the shape starts morphing
  sweepAmplitude: 12,      // metres of lateral sweep during a transition
  convergeShrinkSeconds: 8.0,
  mode: 'swarm',
  showStatus: true,
  showOverlay: true, showTrails: true,

  // rosbridge (RF_Swarm live feed). Deliberately NOT a hardcoded URL: the
  // tunnel is ephemeral (changes every restart) and this repo is public on
  // GitHub Pages, so the live endpoint is passed at runtime via ?ros=<wss url>
  // instead of being committed — see ros-feed.js. No URL = scripted demo only.
  rosbridgeUrl: new URLSearchParams(location.search).get('ros') || '',
  rosbridgeTopic: '/scene_update',
  rosbridgeMsgType: 'std_msgs/String'
};

const CLASSES = {
  controller: { name:'DRONE CONTROLLER', color:'#2dffa0', shape:'remote' },
  enemy:      { name:'ENEMY DRONE',      color:'#ff5252', shape:'drone' },
  walkie:     { name:'WALKIE TALKIE',    color:'#5a82ff', shape:'antenna' },
  jammer:     { name:'RF JAMMER',        color:'#ff4df0', shape:'orb' },
  gcs:        { name:'GROUND STATION',   color:'#39e0ff', shape:'antenna' },
  radar:      { name:'RADAR PING',       color:'#ffd24d', shape:'orb' },
  iot:        { name:'IOT BURST',        color:'#b26bff', shape:'antenna' },
  unknown:    { name:'UNKNOWN EMITTER',  color:'#ff9640', shape:'orb' },
  swarm:      { name:'SWARM DRONE',      color:'#ff5252', shape:'drone' }
};
const CLAIM_COLOR = '#ffb454';
// Models render at world scale, so at 1:20 a 0.5m airframe is ~3cm on screen.
// This multiplier keeps the drones readable as objects rather than specks.
const MODEL_SCALE = 3.4;
const SWARM_COLORS = ['#ff5252','#ffb454','#ffe14d','#2dffa0','#39e0ff','#b26bff'];

const BLOB_EMITTERS = [
  { id:'EMT-007', cls:'controller', confidence:0.79, pos:[ 14,0,-22], vel:[ 0.35,0,-0.20], mode:'blobs' },
  { id:'EMT-013', cls:'enemy',      confidence:0.88, pos:[-30,0,-30], vel:[-1.60,0, 0.90], mode:'blobs' },
  { id:'EMT-021', cls:'walkie',     confidence:0.64, pos:[ 40,0, -8], vel:[ 0.10,0, 0.15], mode:'blobs' },
  { id:'EMT-031', cls:'enemy',      confidence:0.81, pos:[-10,0, 26], vel:[ 0.60,0, 1.20], mode:'blobs' },
  { id:'EMT-042', cls:'jammer',     confidence:0.92, pos:[ 26,0, 18], vel:[-0.40,0,-0.55], mode:'blobs' },
  { id:'EMT-055', cls:'gcs',        confidence:0.71, pos:[-42,0,  6], vel:[ 0.25,0, 0.30], mode:'blobs' },
  { id:'EMT-063', cls:'radar',      confidence:0.58, pos:[  4,0,-44], vel:[ 0.90,0, 0.40], mode:'blobs' },
  { id:'EMT-078', cls:'iot',        confidence:0.45, pos:[-18,0,-10], vel:[ 0.15,0,-0.65], mode:'blobs' }
];

const SWARM_EMITTERS = [];
for (let i=1;i<=6;i++) SWARM_EMITTERS.push({
  id:'DRN-'+i, cls:'swarm', confidence:0.9, colorOverride:SWARM_COLORS[i-1],
  pos:[0,100,0], vel:[0,0,0], driven:true, mode:'swarm'
});

const ALL_EMITTERS = BLOB_EMITTERS.concat(SWARM_EMITTERS);
