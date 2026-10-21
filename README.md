# Smart Band Health-Risk Warning API (Flask + scikit-learn + Vercel)

A Flask web app and JSON REST API for the Smart Band **Machine Learning Mini Project**. It includes a responsive UI where users enter sensor parameters and receive a Random Forest prediction. Team: Faisal Alvi (231202), Zaid Mirkar (231236), and Ibrahim Maniyar (231234). Train and test locally first, then push this project to GitHub and import the repository into Vercel.

> **Research/demo only:** This model predicts the dataset's `alert_flag` label. It is not a clinically validated diagnostic system. Do not use it for medical decisions. The dataset contains synthetic/localized fields and shifted timestamps; do not describe them as newly collected 2026 measurements.

## 1. Project structure

```text
smart_band_flask_api/
├── api/
│   └── index.py
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── app.js
├── app.py
├── feature_engineering.py
├── train_model.py
├── model.joblib                 # created by training; commit this for Vercel
├── training_metrics.json        # created by training
├── requirements.txt
├── vercel.json
├── .python-version
└── .gitignore
```

The CSV is deliberately not included in this project archive. Add your dataset locally as `smart_band_indian_dataset.csv` to train the model. The CSV is ignored by Git to reduce the risk of publishing personal-like identifiers.

## 2. Run locally on Windows

Open PowerShell in this project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `smart_band_indian_dataset(2).csv` into this folder and rename it to:

```text
smart_band_indian_dataset.csv
```

Train the model:

```powershell
python train_model.py
```

This creates `model.joblib` and `training_metrics.json`. Then start Flask:

```powershell
python app.py
```

The API will run at `http://127.0.0.1:5000`.

If PowerShell blocks activation, you can instead run `.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.venv\Scripts\python.exe train_model.py`.

## 3. Test endpoints locally

Health check:

```powershell
curl.exe http://127.0.0.1:5000/health
```

Model metadata:

```powershell
curl.exe http://127.0.0.1:5000/metadata
```

Prediction (PowerShell):

```powershell
$body = @{
  recorded_at = "2026-10-21T09:30:00"
  age_years = 42
  gender = "Female"
  city = "Mumbai"
  state = "Maharashtra"
  heart_rate_bpm = 99
  body_temperature_c = 36.8
  blood_oxygen_percent = 97
  systolic_bp_mmhg = 152
  diastolic_bp_mmhg = 94
  battery_percent = 80
  steps = 7364
  band_removed = "No"
  location_method = "GPS"
  latitude = 19.076
  longitude = 72.8777
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method Post -ContentType "application/json" -Body $body
```

You can also test with Postman: `POST http://127.0.0.1:5000/predict`, Body → raw → JSON. The `/` route returns a sample request.

## 4. Push to GitHub

1. Create an empty GitHub repository, for example `smart-band-health-risk-api`.
2. Make sure `model.joblib` and `training_metrics.json` exist after training. `model.joblib` must be committed because the Vercel function loads it at runtime.
3. Check that the CSV is not staged. It is excluded by `.gitignore`.
4. From this folder:

```powershell
git init
git add .
git status
git commit -m "Add Smart Band health-risk Flask API"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/smart-band-health-risk-api.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username and use the actual repository URL.

## 5. Deploy to Vercel from GitHub

1. Sign in to Vercel and choose **Add New → Project**.
2. Import the GitHub repository.
3. Keep the project root as the repository root. `vercel.json` routes requests to `api/index.py`.
4. Deploy. Once successful, Vercel gives you a URL like `https://your-project.vercel.app`.
5. Test:
   - `https://your-project.vercel.app/`
   - `https://your-project.vercel.app/health`
   - `https://your-project.vercel.app/metadata`
6. Send a POST request to `https://your-project.vercel.app/predict` using the same JSON body as above.

Each new GitHub push normally triggers a new Vercel deployment.

## 6. Important deployment notes

- Keep the Python runtime compatible with the pinned packages. If Vercel's build logs report dependency/runtime incompatibility, use a Python version supported by the current Vercel Python runtime and align `.python-version` and package pins.
- Vercel functions are serverless: don't store uploaded files or mutable state on local disk between requests.
- The serialized scikit-learn model is loaded from the repository at runtime. Keep `model.joblib` in the project root and do not move it unless `app.py` is updated.
- Do not commit the source CSV if it contains contact numbers, names, device IDs, or other sensitive fields. The model file and metrics are sufficient for inference.
- `alert_flag` is the historical target label. If this label was generated from the same measurements being used as features, the model may learn dataset-specific patterns rather than generalizable health risk. Validate carefully before making claims about performance.
- The API currently accepts a single JSON record per request. The model pipeline imputes missing values and ignores unknown categorical levels.

## 7. API response example

```json
{
  "prediction": 1,
  "alert_flag": "Yes",
  "risk_label": "Alert predicted",
  "class_probabilities": {
    "no_alert": 0.18,
    "alert": 0.82
  },
  "disclaimer": "Educational research output only..."
}
```

Probabilities above are illustrative; actual values depend on the trained model and input.
