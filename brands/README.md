# Home Assistant brands submission (WM icon)

Home Assistant renders the **integration-card icon** (Settings → Devices
& Services) from the central [`home-assistant/brands`](https://github.com/home-assistant/brands)
repository, keyed by the integration's `domain` (`spot_price_predictor`).
It does **not** read `custom_components/spot_price_predictor/icon.png` for
that card — so the icon only appears once the asset below is merged into
`home-assistant/brands`.

This is the shared **Watti-matti** WM hexagon badge — the same mark used
by the [Energy Needs Planner](https://github.com/watti-matti/HA-energy-needs-planner)
integration, so the two products read as one family in Home Assistant.

## What's here

```
brands/custom_integrations/spot_price_predictor/
├── icon.png       # 256×256, RGBA, transparent background, trimmed
└── icon@2x.png    # 512×512, RGBA, transparent background, trimmed
```

## How to submit the brands PR

```bash
# 1. Fork + clone the brands repo
gh repo fork home-assistant/brands --clone --remote
cd brands

# 2. Copy the staged assets into place
mkdir -p custom_integrations/spot_price_predictor
cp /path/to/HA-spot-price-predictor/brands/custom_integrations/spot_price_predictor/icon*.png \
   custom_integrations/spot_price_predictor/

# 3. Validate locally (the brands repo ships its own checker)
python3 -m script.hassfest      # or: python3 -m script  (see brands README)

# 4. Branch, commit, push, PR
git checkout -b add-spot-price-predictor-icon
git add custom_integrations/spot_price_predictor
git commit -m "Add Spot Price Predictor (spot_price_predictor) icon"
git push -u origin add-spot-price-predictor-icon
gh pr create --repo home-assistant/brands \
  --title "Add Spot Price Predictor (spot_price_predictor) icon" \
  --body "Custom integration: https://github.com/watti-matti/HA-spot-price-predictor"
```

## Regenerating the assets

`scripts/make_brand_icons.py` regenerates every icon from the one pristine
source badge (`brand/watti-matti_logo.png`, 1024×1024) — flood-fill the
exterior black to transparent → trim → pad square → export 256/512. Pure
downsamples, no upscaling.
