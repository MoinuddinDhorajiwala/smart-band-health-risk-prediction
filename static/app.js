
const form = document.getElementById("predictionForm");
const sampleBtn = document.getElementById("sampleBtn");
const predictBtn = document.getElementById("predictBtn");
const resultCard = document.getElementById("resultCard");
const errorCard = document.getElementById("errorCard");

// Sample Indian smart-band data for testing
const sample = {
  age_years: 42,
  gender: "Female",
  city: "Mumbai",
  state: "Maharashtra",
  recorded_at: "2026-10-21T09:30",
  heart_rate_bpm: 99,
  body_temperature_c: 36.8,
  blood_oxygen_percent: 97,
  systolic_bp_mmhg: 152,
  diastolic_bp_mmhg: 94,
  steps: 7364,
  battery_percent: 80,
  band_removed: "No",
  location_method: "GPS",
  latitude: 19.076,
  longitude: 72.8777
};

// Load sample values into the form
function loadSample() {
  Object.entries(sample).forEach(([name, value]) => {
    const input = form.elements.namedItem(name);

    if (input) {
      input.value = value;
    }
  });

  errorCard.hidden = true;
  resultCard.hidden = true;
}

if (sampleBtn) {
  sampleBtn.addEventListener("click", loadSample);
}

// Convert form inputs into a JSON payload
function buildPayload() {
  const payload = {};

  const numericFields = new Set([
    "age_years",
    "heart_rate_bpm",
    "body_temperature_c",
    "blood_oxygen_percent",
    "systolic_bp_mmhg",
    "diastolic_bp_mmhg",
    "steps",
    "battery_percent",
    "latitude",
    "longitude"
  ]);

  for (const [key, value] of new FormData(form).entries()) {
    const trimmed = value.trim();

    // Skip empty optional values
    if (trimmed === "") {
      continue;
    }

    if (numericFields.has(key)) {
      const numberValue = Number(trimmed);

      if (!Number.isFinite(numberValue)) {
        throw new Error(`Please enter a valid number for ${key}.`);
      }

      payload[key] = numberValue;
    } else {
      payload[key] = trimmed;
    }
  }

  return payload;
}

// Update a probability percentage and progress bar
function setProbability(barId, textId, value) {
  const probability = Math.max(
    0,
    Math.min(1, Number(value ?? 0))
  );

  const percentage = probability * 100;

  const bar = document.getElementById(barId);
  const text = document.getElementById(textId);

  if (bar) {
    bar.style.width = `${percentage}%`;
  }

  if (text) {
    text.textContent = `${percentage.toFixed(1)}%`;
  }
}

// Display the model prediction
function showResult(data) {
  resultCard.hidden = false;
  errorCard.hidden = true;

  const isAlert =
    Number(data.prediction) === 1 ||
    String(data.alert_flag).toLowerCase() === "yes";

  resultCard.classList.toggle("alert", isAlert);

  const resultBadge = document.getElementById("resultBadge");
  const resultSymbol = document.getElementById("resultSymbol");
  const resultLabel = document.getElementById("resultLabel");
  const resultDescription =
    document.getElementById("resultDescription");
  const disclaimerText =
    document.getElementById("disclaimerText");

  if (resultBadge) {
    resultBadge.textContent = isAlert
      ? "ALERT CLASS"
      : "NO-ALERT CLASS";
  }

  if (resultSymbol) {
    resultSymbol.textContent = isAlert ? "!" : "✓";
  }

  if (resultLabel) {
    resultLabel.textContent =
      data.risk_label ||
      (isAlert ? "Alert predicted" : "No alert predicted");
  }

  if (resultDescription) {
    resultDescription.textContent = isAlert
      ? "The model classified this input as matching the alert class in the training data."
      : "The model classified this input as matching the no-alert class in the training data.";
  }

  const probabilities = data.class_probabilities || {};

  setProbability(
    "noAlertBar",
    "noAlertText",
    probabilities.no_alert
  );

  setProbability(
    "alertBar",
    "alertText",
    probabilities.alert
  );

  if (disclaimerText) {
    disclaimerText.textContent =
      data.disclaimer ||
      "Educational research output only. This is not a medical diagnosis.";
  }

  resultCard.scrollIntoView({
    behavior: "smooth",
    block: "center"
  });
}

// Display errors in the page
function showError(message) {
  errorCard.textContent = message;
  errorCard.hidden = false;
}

// Submit the form and request a prediction from Flask
if (form) {
  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    errorCard.hidden = true;
    resultCard.hidden = true;

    predictBtn.disabled = true;
    predictBtn.innerHTML =
      "<span>Running model…</span><b>◌</b>";

    try {
      const payload = buildPayload();

      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });

      let data;

      try {
        data = await response.json();
      } catch {
        throw new Error(
          "The server returned an invalid response. Check the Flask API."
        );
      }

      if (!response.ok) {
        throw new Error(
          data.error || `Request failed (${response.status})`
        );
      }

      showResult(data);
    } catch (error) {
      showError(
        `Prediction failed: ${error.message}. ` +
        "Check that the Flask server is running and the model file exists."
      );
    } finally {
      predictBtn.disabled = false;
      predictBtn.innerHTML =
        "<span>Predict alert pattern</span><b>↗</b>";
    }
  });
}
