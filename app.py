import json
import os

import pickle
import pandas as pd
from flask import Flask, Response, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "linear.pkl")

app = Flask(__name__)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

CURRENCY = "$"  # change to "₹" if your training data is in rupees

# Order MUST match the model's training columns
FEATURES = list(model.feature_names_in_)

# UI config for each input (label, icon, min, max, step, default)
FIELDS = [
    {"name": "Square_Footage", "label": "Square Footage", "icon": "📐", "min": 300, "max": 10000, "step": 10, "value": 2000},
    {"name": "Num_Bedrooms", "label": "Bedrooms", "icon": "🛏️", "min": 1, "max": 10, "step": 1, "value": 3},
    {"name": "Num_Bathrooms", "label": "Bathrooms", "icon": "🛁", "min": 1, "max": 8, "step": 1, "value": 2},
    {"name": "Year_Built", "label": "Year Built", "icon": "🏗️", "min": 1900, "max": 2026, "step": 1, "value": 2005},
    {"name": "Lot_Size", "label": "Lot Size", "icon": "🌳", "min": 0.1, "max": 10, "step": 0.01, "value": 1.5},
    {"name": "Garage_Size", "label": "Garage Size (cars)", "icon": "🚗", "min": 0, "max": 6, "step": 1, "value": 1},
    {"name": "Neighborhood_Quality", "label": "Neighborhood Quality", "icon": "🏙️", "min": 1, "max": 10, "step": 1, "value": 5},
]

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HomeWorth · House Price Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap">
<style>
  :root{
    --bg1:#0f0c29; --bg2:#302b63; --bg3:#24243e;
    --accent:#7c5cff; --accent2:#00d4ff; --accent3:#ff6ec7;
    --glass:rgba(255,255,255,.08); --border:rgba(255,255,255,.18);
    --text:#f4f4ff; --muted:#b9b6e6;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{
    font-family:'Poppins',system-ui,sans-serif; color:var(--text);
    min-height:100vh; display:flex; align-items:center; justify-content:center;
    padding:32px 16px; overflow-x:hidden;
    background:linear-gradient(135deg,var(--bg1),var(--bg2),var(--bg3));
    background-size:300% 300%; animation:bgShift 18s ease infinite;
  }
  @keyframes bgShift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}

  /* floating blobs */
  .blob{position:fixed;border-radius:50%;filter:blur(70px);opacity:.55;z-index:0;animation:float 14s ease-in-out infinite}
  .b1{width:340px;height:340px;background:var(--accent);top:-80px;left:-80px}
  .b2{width:300px;height:300px;background:var(--accent2);bottom:-90px;right:-60px;animation-delay:-5s}
  .b3{width:240px;height:240px;background:var(--accent3);top:45%;left:60%;animation-delay:-9s;opacity:.35}
  @keyframes float{0%,100%{transform:translate(0,0) scale(1)}33%{transform:translate(40px,-50px) scale(1.12)}66%{transform:translate(-30px,40px) scale(.94)}}

  .card{
    position:relative; z-index:1; width:100%; max-width:760px;
    background:var(--glass); border:1px solid var(--border); border-radius:28px;
    backdrop-filter:blur(22px); -webkit-backdrop-filter:blur(22px);
    padding:40px 36px; box-shadow:0 30px 80px rgba(0,0,0,.45);
    animation:rise .9s cubic-bezier(.2,.8,.2,1) both;
  }
  @keyframes rise{from{opacity:0;transform:translateY(40px) scale(.97)}to{opacity:1;transform:none}}

  header{text-align:center;margin-bottom:30px}
  .logo{font-size:54px;display:inline-block;animation:bob 3s ease-in-out infinite}
  @keyframes bob{0%,100%{transform:translateY(0) rotate(-4deg)}50%{transform:translateY(-12px) rotate(4deg)}}
  h1{font-size:2.1rem;font-weight:800;margin-top:6px;
     background:linear-gradient(90deg,var(--accent2),var(--accent),var(--accent3));
     -webkit-background-clip:text;background-clip:text;color:transparent;
     background-size:200% auto;animation:shine 5s linear infinite}
  @keyframes shine{to{background-position:200% center}}
  header p{color:var(--muted);font-size:.95rem;margin-top:6px}

  .grid{display:grid;grid-template-columns:repeat(2,1fr);gap:18px}
  .field{
    background:rgba(255,255,255,.06); border:1px solid var(--border); border-radius:16px;
    padding:14px 16px; transition:transform .25s, box-shadow .25s, border-color .25s;
    opacity:0; animation:fadeUp .6s ease forwards;
  }
  .field:hover,.field:focus-within{transform:translateY(-4px);border-color:var(--accent2);box-shadow:0 10px 26px rgba(0,212,255,.18)}
  .field.wide{grid-column:1/-1}
  @keyframes fadeUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:none}}
  label{display:flex;align-items:center;gap:8px;font-size:.82rem;font-weight:500;color:var(--muted);margin-bottom:8px}
  label .ico{font-size:1.15rem}
  input[type=number]{
    width:100%; background:transparent; border:none; outline:none;
    color:var(--text); font:600 1.2rem 'Poppins',sans-serif; padding:2px 0;
  }
  input[type=range]{width:100%;margin-top:10px;accent-color:var(--accent);cursor:pointer;height:4px}

  button{
    width:100%; margin-top:28px; padding:16px; border:none; border-radius:16px; cursor:pointer;
    font:700 1.05rem 'Poppins',sans-serif; color:#fff; letter-spacing:.4px;
    background:linear-gradient(90deg,var(--accent),var(--accent2),var(--accent3));
    background-size:200% auto; transition:transform .2s, box-shadow .3s, background-position .5s;
    box-shadow:0 12px 30px rgba(124,92,255,.45); position:relative; overflow:hidden;
  }
  button:hover{transform:translateY(-3px);background-position:right center;box-shadow:0 18px 40px rgba(0,212,255,.45)}
  button:active{transform:scale(.98)}
  button.loading::after{
    content:"";position:absolute;inset:0;
    background:linear-gradient(110deg,transparent 30%,rgba(255,255,255,.35) 50%,transparent 70%);
    animation:sweep 1s linear infinite;
  }
  @keyframes sweep{from{transform:translateX(-100%)}to{transform:translateX(100%)}}

  .result{
    margin-top:26px; text-align:center; padding:26px 20px; border-radius:20px;
    background:linear-gradient(135deg,rgba(124,92,255,.25),rgba(0,212,255,.18));
    border:1px solid var(--border); display:none;
  }
  .result.show{display:block;animation:pop .7s cubic-bezier(.2,1.4,.4,1) both}
  @keyframes pop{from{opacity:0;transform:scale(.8)}to{opacity:1;transform:scale(1)}}
  .result small{color:var(--muted);letter-spacing:2px;text-transform:uppercase;font-size:.75rem}
  .price{font-size:2.8rem;font-weight:800;margin-top:6px;
         background:linear-gradient(90deg,#fff,var(--accent2));-webkit-background-clip:text;background-clip:text;color:transparent}
  .error{color:#ff8a9b;font-weight:500}

  footer{text-align:center;color:var(--muted);font-size:.75rem;margin-top:22px;opacity:.8}

  @media (max-width:600px){
    .card{padding:28px 20px} .grid{grid-template-columns:1fr} h1{font-size:1.6rem} .price{font-size:2.2rem}
  }
</style>
</head>
<body>
  <div class="blob b1"></div><div class="blob b2"></div><div class="blob b3"></div>

  <main class="card">
    <header>
      <div class="logo">🏡</div>
      <h1>HomeWorth</h1>
      <p>Enter your property details and get an instant AI price estimate</p>
    </header>

    <form id="form" autocomplete="off">
      <div class="grid" id="grid"></div>
      <button type="submit" id="btn">✨ Predict Price</button>
    </form>

    <div class="result" id="result">
      <small>Estimated Value</small>
      <div class="price" id="price">—</div>
    </div>

    <footer>Powered by Linear Regression · Flask · Render</footer>
  </main>

<script>
  const FIELDS = __FIELDS__;
  const CURRENCY = "__CURRENCY__";
  const grid = document.getElementById('grid');

  FIELDS.forEach((f, i) => {
    const div = document.createElement('div');
    div.className = 'field' + (i === FIELDS.length - 1 && FIELDS.length % 2 === 1 ? ' wide' : '');
    div.style.animationDelay = (0.15 + i * 0.08) + 's';
    div.innerHTML = `
      <label for="${f.name}"><span class="ico">${f.icon}</span>${f.label}</label>
      <input type="number" id="${f.name}" name="${f.name}" min="${f.min}" max="${f.max}" step="${f.step}" value="${f.value}" required>
      <input type="range" data-for="${f.name}" min="${f.min}" max="${f.max}" step="${f.step}" value="${f.value}">`;
    grid.appendChild(div);

    const num = div.querySelector('input[type=number]');
    const rng = div.querySelector('input[type=range]');
    num.addEventListener('input', () => rng.value = num.value);
    rng.addEventListener('input', () => num.value = rng.value);
  });

  function countUp(el, target) {
    const duration = 1400, start = performance.now();
    const fmt = n => CURRENCY + Math.round(n).toLocaleString('en-US');
    (function frame(now) {
      const t = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      el.textContent = fmt(target * eased);
      if (t < 1) requestAnimationFrame(frame);
    })(start);
  }

  document.getElementById('form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('btn');
    const box = document.getElementById('result');
    const price = document.getElementById('price');
    const payload = {};
    FIELDS.forEach(f => payload[f.name] = parseFloat(document.getElementById(f.name).value));

    btn.classList.add('loading'); btn.textContent = 'Calculating…';
    try {
      const res = await fetch('/predict', {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
      });
      const data = await res.json();
      box.classList.remove('show'); void box.offsetWidth; // restart animation
      box.classList.add('show');
      if (data.error) { price.innerHTML = '<span class="error">' + data.error + '</span>'; }
      else { price.classList.remove('error'); countUp(price, data.prediction); }
      box.scrollIntoView({behavior: 'smooth', block: 'nearest'});
    } catch (err) {
      box.classList.add('show');
      price.innerHTML = '<span class="error">Server error. Try again.</span>';
    } finally {
      btn.classList.remove('loading'); btn.textContent = '✨ Predict Price';
    }
  });
</script>
</body>
</html>
"""


@app.route("/")
def home():
    html = PAGE.replace("__FIELDS__", json.dumps(FIELDS)).replace("__CURRENCY__", CURRENCY)
    return Response(html, mimetype="text/html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        row = {name: float(data[name]) for name in FEATURES}
        X = pd.DataFrame([row], columns=FEATURES)
        pred = float(model.predict(X)[0])
        return jsonify({"prediction": max(pred, 0)})
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e.args[0]}"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Please enter valid numbers in every field."}), 400
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Prediction failed: {e}"}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
