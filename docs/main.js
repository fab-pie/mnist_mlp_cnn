// --- Importer les modules exportés (default) depuis les dossiers des modèles ---
// mnist_convnet and mnist_mlp live at repository root; main.js is in /Webapp
import cnnModule from '../mnist_convnet/mnist_convnet.js';
import mlpModule from '../mnist_mlp/mnist_mlp.js';

// --- Éléments HTML ---
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clearBtn");
const modelSelect = document.getElementById("modelSelect");
const resultsDiv = document.getElementById("results");
const loader = document.getElementById("loader");

// --- Canvas initialisation ---
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

let drawing = false;
canvas.addEventListener("mousedown", () => drawing = true);
canvas.addEventListener("mouseup", () => drawing = false);
canvas.addEventListener("mouseleave", () => drawing = false);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener('touchstart', (e) => { drawing = true; draw(e); });
canvas.addEventListener('touchend', () => { drawing = false; });
canvas.addEventListener('touchmove', draw);

function draw(e) {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  const clientY = e.touches ? e.touches[0].clientY : e.clientY;
  const x = clientX - rect.left;
  const y = clientY - rect.top;

  ctx.fillStyle = "black";
  ctx.beginPath();
  ctx.arc(x, y, 10, 0, Math.PI * 2);
  ctx.fill();

  // Débounced prediction handled elsewhere
  schedulePredict();
}

clearBtn.addEventListener("click", () => {
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  resultsDiv.innerHTML = "";
});

// --- Variables modèles ---
let cnnModel, mlpModel;
let modelsReady = false;
let cnnRunner = null;
let mlpRunner = null;
// persistent stats per model saved in localStorage
const STORAGE_KEY = 'mnist_model_stats_v1';
let modelStats = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
if (!modelStats.cnn) modelStats.cnn = { correct: 0, total: 0 };
if (!modelStats.mlp) modelStats.mlp = { correct: 0, total: 0 };

function saveStats() { localStorage.setItem(STORAGE_KEY, JSON.stringify(modelStats)); }
function updateAccuracyDisplay() {
  const cur = modelSelect.value;
  const curStats = modelStats[cur] || { correct: 0, total: 0 };
  const cnnAcc = modelStats.cnn.total ? (modelStats.cnn.correct / modelStats.cnn.total * 100).toFixed(2) + '%' : 'N/A';
  const mlpAcc = modelStats.mlp.total ? (modelStats.mlp.correct / modelStats.mlp.total * 100).toFixed(2) + '%' : 'N/A';
  const curAcc = curStats.total ? (curStats.correct / curStats.total * 100).toFixed(2) + '%' : 'N/A';
  const accEl = document.getElementById('accuracyDisplay');
  if (accEl) accEl.textContent = `Model ${cur.toUpperCase()} — accuracy: ${curAcc} (CNN: ${cnnAcc}, MLP: ${mlpAcc})`;
}

function recordFeedback(isCorrect) {
  const m = modelSelect.value;
  if (!modelStats[m]) modelStats[m] = { correct: 0, total: 0 };
  modelStats[m].total += 1;
  if (isCorrect) modelStats[m].correct += 1;
  saveStats();
  updateAccuracyDisplay();
}

// --- Charger les modèles WebGPU ---
async function loadModels() {
  try {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();

    // Charger CNN
    // modules export default an object with .load
    const cnnLoadedModule = cnnModule;
    const mlpLoadedModule = mlpModule;

  // note: files in the repo use the .webgpu.safetensors extension
    // fetch the safetensor files ourselves so we can validate the response
    async function fetchSafetensor(path) {
      const resp = await fetch(path);
      if (!resp.ok) throw new Error(`Failed to fetch ${path}: ${resp.status} ${resp.statusText}`);
      const ab = await resp.arrayBuffer();
      // quick heuristic: if the response starts with '<' it's probably HTML (an error page)
      const firstChar = new TextDecoder().decode(new Uint8Array(ab, 0, Math.min(1, ab.byteLength)));
      if (firstChar === '<') {
        // also provide a short text preview to help debugging
        const txt = new TextDecoder().decode(new Uint8Array(ab).subarray(0, Math.min(ab.byteLength, 512)));
        throw new Error(`Received HTML instead of safetensor for ${path}: ${txt.slice(0,200)}`);
      }
      return new Uint8Array(ab);
    }

  // weights are stored at repo root next to the model JS files
  const cnnWeights = await fetchSafetensor("../mnist_convnet/mnist_convnet.webgpu.safetensors");
  const mlpWeights = await fetchSafetensor("../mnist_mlp/mnist_mlp.webgpu.safetensors");

    // Use setupNet directly (module exposes setupNet) so we don't re-fetch inside the module
    cnnRunner = await cnnLoadedModule.setupNet(device, cnnWeights);
    mlpRunner = await mlpLoadedModule.setupNet(device, mlpWeights);

    // expose runners for manual debugging in the console
    window._cnnRunner = cnnRunner;
    window._mlpRunner = mlpRunner;

    // show weight sizes to help verify what's loaded
    const cnnKB = Math.round((cnnWeights.byteLength || cnnWeights.length) / 1024);
    const mlpKB = Math.round((mlpWeights.byteLength || mlpWeights.length) / 1024);
    loader.textContent = `Models loaded — CNN: ${cnnKB} KB, MLP: ${mlpKB} KB`;

    // display active model in the results area
    resultsDiv.innerHTML = `<div><strong>Modèle actif:</strong> ${modelSelect.value.toUpperCase()}</div>`;

    // when the user switches model, update the displayed active model and re-run prediction
    modelSelect.addEventListener('change', () => {
      resultsDiv.querySelector('strong').textContent = 'Modèle actif:';
      resultsDiv.firstChild && (resultsDiv.firstChild.nextSibling.textContent = ` ${modelSelect.value.toUpperCase()}`);
      if (modelsReady) schedulePredict();
      updateAccuracyDisplay();
    });

    // cnnRunner and mlpRunner are async functions to call for inference
    cnnModel = !!cnnRunner;
    mlpModel = !!mlpRunner;

    modelsReady = true;
    loader.style.display = "none";
    console.log("Models loaded!");
  updateAccuracyDisplay();
  } catch (err) {
    console.error("Erreur lors du chargement des modèles:", err);
    loader.textContent = "Erreur de chargement des modèles.";
  }
}

loadModels();

// --- Transformer le canvas en tableau 28x28 normalisé ---
function getImageData() {
  const tmpCanvas = document.createElement("canvas");
  tmpCanvas.width = 28;
  tmpCanvas.height = 28;
  const tmpCtx = tmpCanvas.getContext("2d");
  tmpCtx.drawImage(canvas, 0, 0, 28, 28);
  const imgData = tmpCtx.getImageData(0, 0, 28, 28);
  const data = [];
  for (let i = 0; i < imgData.data.length; i += 4) {
    const val = imgData.data[i]; // R=G=B
    // map [0..255] -> [-1..1], invert so drawn black (near 0) becomes +1
    data.push((255 - val) * 2 / 255 - 1);
  }
  return data;
}

// --- Prédiction ---
async function predict() {
  if (!modelsReady) return;

  const inputData = getImageData();
  const modelChoice = modelSelect.value;
  let result;

  try {
    const typed = new Float32Array(inputData.length);
    typed.set(inputData);

    const t0 = performance.now();
    let out;
    if (modelChoice === "cnn") out = await cnnRunner(typed);
    else out = await mlpRunner(typed);
    const t1 = performance.now();

    // out is usually an array of Float32Array outputs
    const logits = (Array.isArray(out) ? out[0] : out);
    const probs = softmax(Array.from(logits));

    // Render results
    const bestIdx = probs.indexOf(Math.max(...probs));
    let html = `<h3>Prédiction : ${bestIdx} (${(probs[bestIdx]*100).toFixed(2)}%)</h3>`;
    html += `<div>Temps inférence : ${(t1 - t0).toFixed(2)} ms</div>`;
    // render probabilities as vertical bars that grow upwards, all digits in one row
    html += '<h4>Probabilités</h4>';
    // make the frame larger and force single-line layout; allow horizontal scroll on small screens
    html += '<div id="probs" style="display:flex;gap:14px;flex-wrap:nowrap;justify-content:center;align-items:flex-end;padding:12px 6px;overflow-x:auto;max-width:100%;">';
    probs.forEach((p,i)=>{
      const pct = (p*100).toFixed(2);
      const fillPct = Math.max(4, Math.round(p * 100)); // ensure tiny bars still visible
      const color = (i===bestIdx) ? '#4caf50' : '#2196f3';
      // each cell: taller bar area so bars stack upwards; increased width for readability
      html += `
        <div style="width:80px;text-align:center;font-size:12px;">
          <div style="height:120px;display:flex;align-items:flex-end;justify-content:center;background:transparent;padding-bottom:6px;">
            <div style="width:42px;height:${fillPct}%;background:${color};border-radius:6px 6px 0 0;box-shadow: 0 2px 6px rgba(0,0,0,0.12);transition:height 250ms ease;"></div>
          </div>
          <div style="font-weight:700;margin-top:8px">${i}</div>
          <div style="font-size:11px;color:#222">${pct}%</div>
        </div>`;
    });
    html += '</div>';
    resultsDiv.innerHTML = html;

    // ensure the probabilities container is scrolled to the left so digits 0 and 1 are visible
    const probsEl = document.getElementById('probs');
    if (probsEl) {
      // small timeout so the browser lays out the new content first
      setTimeout(() => { try { probsEl.scrollLeft = 0; } catch (e) {} }, 20);
    }

    // feedback UI inserted under probabilities: ✅ / ❌ + accuracy display
    const feedbackHtml = `
      <div id="feedback" style="margin-top:8px;display:flex;align-items:center;gap:12px;">
        <button id="btnCorrect" title="C'est correct" style="background:#4caf50;color:white;border:none;padding:8px 12px;border-radius:6px;cursor:pointer;font-size:16px;">✅</button>
        <button id="btnWrong" title="C'est incorrect" style="background:#f44336;color:white;border:none;padding:8px 12px;border-radius:6px;cursor:pointer;font-size:16px;">❌</button>
        <div id="accuracyDisplay" style="font-size:14px;color:#333;margin-left:8px;">Loading accuracy...</div>
      </div>
    `;
    resultsDiv.insertAdjacentHTML('beforeend', feedbackHtml);

    // attach handlers for feedback buttons
    const btnC = document.getElementById('btnCorrect');
    const btnW = document.getElementById('btnWrong');
    if (btnC) btnC.addEventListener('click', () => recordFeedback(true));
    if (btnW) btnW.addEventListener('click', () => recordFeedback(false));

    // update accuracy display for current model
    updateAccuracyDisplay();
  } catch (err) {
    console.error("Erreur pendant la prédiction:", err);
  }
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const s = exps.reduce((a,b)=>a+b,0);
  return exps.map(x => x / s);
}

// debounce predictions while drawing
let predictTimeout = null;
function schedulePredict() {
  if (predictTimeout) clearTimeout(predictTimeout);
  predictTimeout = setTimeout(() => { predict(); predictTimeout = null; }, 150);
}
