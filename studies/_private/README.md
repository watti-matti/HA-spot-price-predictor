# `studies/_private/` — local-only directory

This directory is **gitignored**. Anything placed here stays on your
machine and never enters the repository.

Intended contents:

- `home-assistant_v2.db` (or any HA recorder snapshot)
- `household_profile.json` — derived shape statistics for the
  `PV_adjusted_price` study
- `sensor_map.json` — mapping from canonical names to your HA
  entity IDs
- CSV exports under `_raw/`

The privacy contract is defined in
[`../docs/household_profile_schema.md`](../docs/household_profile_schema.md).

Public code in this repo MUST NOT hardcode paths under this
directory. Always parameterise via CLI argument or environment
variable, with a synthetic fallback. The README you are reading is
the only file under `_private/` that is committed (so the directory
exists in a fresh clone).
