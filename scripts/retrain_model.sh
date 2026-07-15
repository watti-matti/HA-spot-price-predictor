#!/usr/bin/env bash
# ============================================================
#  HA Spot Price Predictor - one-shot PC retraining script
#
#  Retrains EVERYTHING on your PC (fast) so your Home Assistant
#  host (e.g. a Raspberry Pi) never has to. It refits both:
#    * the base hourly + D(k) duration model, and
#    * the pipeline layers (L1 seasonal, L2-L4 spike, solar),
#  then stages the 4 updated JSON artifacts ready to copy to HA.
#
#  Run: ./scripts/retrain_model.sh   (it finds the repo from its
#  own path, so keep it in the scripts/ folder).
#  Requires Python 3.11+ and the repo checked out on this PC.
# ============================================================

set -euo pipefail

# ======================= EDIT ME ===========================
# Fingrid API key (free, https://data.fingrid.fi). OPTIONAL:
# leave EMPTY to train without nuclear features and skip the
# solar sub-model. The model still works fully without it.
FINGRID_API_KEY=""

# Region config file under config/regions/<REGION>.yaml
REGION="finland"

# Years of price history to train on (4 = recommended).
YEARS="4"

# Python launcher. Use a venv if you have one, e.g.
#   PYTHON="/path/to/venv/bin/python"
PYTHON="python3"

# Install/upgrade Python dependencies first? 1=yes 0=no
INSTALL_DEPS=1

# Run the training unit tests at the end? 1=yes 0=no
RUN_TESTS=1
# ===========================================================

cd "$(dirname "$0")/.."
echo "Repo root: $(pwd)"
echo

FK=()
if [ -n "$FINGRID_API_KEY" ]; then
  FK=(--fingrid-key "$FINGRID_API_KEY")
else
  echo "[note] No Fingrid key - nuclear features off, solar sub-model skipped."
fi
echo

fail() { echo; echo "*** Retraining FAILED - see error above. Do NOT copy output/ha_deploy to HA until a clean run completes. ***"; exit 1; }
trap fail ERR

if [ "$INSTALL_DEPS" = "1" ]; then
  echo "=== [1/5] Installing Python dependencies ==="
  "$PYTHON" -m pip install -r requirements.txt scikit-learn pytest
  echo
fi

echo "=== [2/5] Training base model: fetch data + hourly Ridge + duration D(k) ==="
# Auto-retry: prices + each weather location are cached on disk, so a
# transient timeout only re-fetches what failed - not from scratch.
attempt=0
until "$PYTHON" -m src.train_model --region "$REGION" "${FK[@]}" --years "$YEARS"; do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge 3 ]; then echo "training failed after $attempt attempts"; fail; fi
  echo "[retry] training failed - attempt $attempt of 3. Cached data reused; retrying in 20s..."
  sleep 20
done
echo

echo "=== [3/5] Retraining pipeline layers: seasonal + spike + solar ==="
# Run retrain.py as a FILE (not -m): the package __init__ imports
# Home Assistant, which is not installed on a training PC.
"$PYTHON" custom_components/spot_price_predictor/retrain.py --layers seasonal spike solar "${FK[@]}"
echo

echo "=== [4/5] Staging updated artifacts ==="
DATA="custom_components/spot_price_predictor/data"
cp -f "output/model_coefs.json" "$DATA/model_coefs_default.json"
mkdir -p "output/ha_deploy"
cp -f "$DATA/model_coefs_default.json"          "output/ha_deploy/"
cp -f "$DATA/seasonal_components_default.json"  "output/ha_deploy/"
cp -f "$DATA/spike_model_default.json"          "output/ha_deploy/"
cp -f "$DATA/solar_submodel_default.json"       "output/ha_deploy/" 2>/dev/null || true
echo "Updated artifacts staged in: $(pwd)/output/ha_deploy"
echo

if [ "$RUN_TESTS" = "1" ]; then
  echo "=== [5/5] Running training tests ==="
  "$PYTHON" -m pytest tests/test_training.py -q
  echo
fi

trap - ERR
cat <<EOF
============================================================
 DONE. To update your Home Assistant host:

   Copy the 3-4 JSON files from
       $(pwd)/output/ha_deploy/
   to your HA host at
       <config>/custom_components/spot_price_predictor/data/
   then restart Home Assistant (or reload the integration).

 Tip: for the base model only you can skip the restart and use
 the spot_price_predictor.upload_coefficients service with
 file_path set to model_coefs_default.json on the HA host.
============================================================
EOF
