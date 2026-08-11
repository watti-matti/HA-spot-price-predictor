# Tekninen toteutus — HA Spot Price Predictor (v2.11.0)

> ## ⚠️ TÄMÄ DOKUMENTTI ON VANHENTUNUT (päivitetty viimeksi v2.11.0, 2026-05-20)
>
> Toimitettava koodi on edennyt yhdeksän julkaisua eteenpäin. Tämä
> dokumentti kuvaa mallin **ennen** kahta merkittävintä korjausta, eikä
> sen kuvausta pidä käyttää:
>
> - **L2-piirteet.** Tässä kuvatut `Y_se1` / `Y_se3` / `Y_ee` ovat
>   *saman tunnin* naapurihintoja. FI, SE1, SE3 ja EE selviävät samassa
>   day-ahead-huutokaupassa, joten niitä ei voi havaita ennen ennustettavaa
>   kohdetta. Tämä vuoto poistettiin v2.17.0:ssa — piirteet ovat nyt
>   `Y_se1_lag168` / `Y_se3_lag168` / `Y_ee_lag168`, ja mukana on myös
>   `is_holiday`. Piirteitä on yhdeksän, ei kahdeksaa.
> - **Auringon etumerkki.** v2.16.0 asetti nollarajakustannuspakotteen:
>   tuulen ja auringon kertoimet ovat aina ≤ 0.
> - **Fysiikkapiirteiden kausitasoitus.** v2.15.0 korvasi paikallisen
>   keskiarvokeskityksen artefaktin `physics_seasonal`-komponenteilla.
> - **Skriptien nimet ja rivinumerot** viittaavat poistettuihin
>   tiedostoihin (esim. `studies/v253_solar_submodel.py`).
>
> **Käytä toistaiseksi englanninkielistä [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md):ta**,
> joka on ajan tasalla. Tämän dokumentin päivitys on kirjattu
> jatkotyöksi (docs/BACKLOG.md).


Suomalaisen kuluttajan sähkön spot- ja kuluttajahinnan sekä D(k)-kestokäyrien ennustaminen Home Assistantiin. Tuottaa 170 tunnin spot/kuluttajahinnan pisteen ennusteen, P5/P25/P50/P75/P95-viuhkavyöt ja 7 vrk:n halpa/kallis-kestokäyrät nelitasoisesta ennustusputkesta. Tämä opas kuvaa vain sen mitä toimitettava koodi todella tekee.

## Arkkitehtuuri

Integraatiossa on kaksi vaihetta:

- **Päättely** ajetaan Home Assistantin sisällä. `SpotPriceCoordinator` ([`coordinator.py`](custom_components/spot_price_predictor/coordinator.py)) ohjaa jaksottaisen päivityskierroksen: hakee säädata- ja hintavirrat, kutsuu `Pipeline`-luokkaa ([`pipeline.py`](custom_components/spot_price_predictor/pipeline.py)) nelitasoiseen ennustukseen, rakentaa tunti- ja päiväkohtaiset attribuutit ja työntää ne sensorientiteeteille ([`sensor.py`](custom_components/spot_price_predictor/sensor.py)).
- **Uudelleenkoulutus** on pyynnöstä tehtävä `data/`-kansion kolmen JSON-artefaktin uudelleensovitus, joka julkaistaan palveluna `spot_price_predictor.retrain_models`. Sovitusskriptit (`studies/build_seasonal_components.py`, `studies/v2513_layer4_spike_model.py`, `studies/v253_solar_submodel.py`) lukevat välimuistissa olevia parquet-tiedostoja ja kirjoittavat artefaktit atomisesti; koordinaattori lataa ne uudelleen seuraavalla kierroksella.

```
Open-Meteo  ──┐
Sahkotin    ──┼──> SpotPriceCoordinator
Elpriset    ──┤      ├── rakentaa 170 ennusterivin matriisin
Elering     ──┤      ├── kutsuu Pipeline.compute_forecast → spot, kuluttajahinta, P5..P95 per rivi
Fingrid     ──┤      ├── laskee päiväkohtaiset D(k)-käyrät (4 × 24 taulukkoa per päivä)
Nord Pool UMM ┘      ├── (valinnainen) PV-tietoinen rikastus rivikohtaisesti + PV-D(k) per päivä
                     └── (valinnainen) DtACI-kalibroija kietoo D(k)-käyrät vöillä
                            │
                            ▼
                  sensor.spot_price_predictor_*
```

Putki lataa kolme jäädytettyä artefaktia `custom_components/spot_price_predictor/data/`-kansiosta:

| Artefakti | Lataus kohteeseen | Sisältö |
|---|---|---|
| `seasonal_components_default.json` | `Pipeline._seasonal_artifact` | Per-sarja additiiviset komponentit (tunti + päivä + viikko) FI-spotille ja lämpötilalle. |
| `spike_model_default.json` | `Pipeline._spike_artifact` | L2 `ridge_coef`, L3 `ar1_phi`, L4 `gpd_left` / `gpd_right` -häntäparametrit, Normaalirungon `stats.eta_train_mean` / `stats.eta_train_sigma`. |
| `solar_submodel_default.json` | (vain PV-tietoinen polku) | Selkeän taivaan × pilvisyys -aurinkotuotantomalli. |

Pysyvä kalibroijan tila kansiossa `<config>/.storage/spot_price_predictor_pipeline/` (`hourly_bias.json`, `hourly_fan_chart.json`, `refit_monitor.json`). Päivityksen yhteydessä vanhempaa versiota seuraava `.storage/spot_price_predictor_v26/`-hakemisto nimetään automaattisesti uudelleen, jotta kertynyt biaskorjaajan historia säilyy.

## Nelitasoinen ennustusputki

Julkinen sisääntulo: `Pipeline.compute_forecast(timestamps, wind, solar, temp, recent_fi_residuals=None, enable_fan_chart=True)` ([`pipeline.py:318`](custom_components/spot_price_predictor/pipeline.py:318)). Jokaisella ennustetunnilla h:

```
mean(h)  = L1 seasonal_fi(h)
         + L2 ridge(h)
         + L3 φ^h · η(t₀−1)
mean(h)  = softplus_floor(mean(h), pohja = −5 EUR/MWh)
mean(h) -= hourly_bias_corrector.bias_estimate   # vain lämmittelyn jälkeen

P{5,25,50,75,95}_eur_mwh(h) ← 500 näytettä sekoituksesta (Normaalirunko + GPD-hännät)
                              keskitettynä arvoon mean(h)
```

### L1 — Kausivaihteludekompositio

`Pipeline._seasonal_fi` ja `Pipeline._deseasonalize_input` ([`pipeline.py:200-216`](custom_components/spot_price_predictor/pipeline.py:200)) lukevat additiiviset tunti + päivä + viikko -komponentit tiedostosta `seasonal_components_default.json` ja vähentävät ne FI-hinnasta ja lämpötilasta tuottaen kausitasoitetut residuaalit. Tuulta ja aurinkoa ei kausitasoiteta L1:n kautta — ne keskitetään paikallisesti (keskiarvon vähennys) ennen Ridge-regressioon syöttämistä.

### L2 — Kausivaihtelusta puhdistetun osan Ridge-regressio

Toimitettava `data/spike_model_default.json` sisältää kanonisen
piirrejärjestyksen `ridge_features`-kentässään; putki lukee sen
rakennusvaiheessa ja muodostaa suunnittelumatriisin tässä
järjestyksessä. Varakenttänä on `RIDGE_FEATURES`-vakio
([`pipeline.py:62-77`](custom_components/spot_price_predictor/pipeline.py:62)).

| # | Piirre | Rakennetaan `_build_features`:ssä ([`pipeline.py:220-275`](custom_components/spot_price_predictor/pipeline.py:220)) | Määritelmä |
|---|---|---|---|
| 1 | `intercept` | `np.ones(n)` | vakio 1 |
| 2 | `Y_fi_lag168` | välitetään kutsujalta `recent_fi_residuals["lag168"]`-avaimella | Kausitasoitettu FI-residuaali 7 päivää aiemmin. Koordinaattori välittää tällä hetkellä nollia (kylmäkäynnistys), koska liukuva ennustushistoria on alle 7 päivää syvä. |
| 3 | `is_workday` | `Pipeline._is_workday` — `weekday < 5` | binaari {0, 1} |
| 4 | `Y_sigmoid_wind_rho` | `_sigmoid_turbine_rho` ([`pipeline.py:81-87`](custom_components/spot_price_predictor/pipeline.py:81)), sitten keskitetään paikallisesti | `σ((tuuli − 7,5) / 1,5) × ρ(T) / 1,225` |
| 5 | `Y_solar_effective` | `_solar_effective` ([`pipeline.py:90-96`](custom_components/spot_price_predictor/pipeline.py:90)), sitten keskitetään paikallisesti | `GHI × (1 − 0,004 · max(0, T_cell − 25))`, `T_cell = T + 0,03 · GHI` |
| 6 | `Y_temp` | `_deseasonalize_input("temp", …)` | Kausitasoitettu lämpötila |
| 7 | `Y_se1` | `_deseasonalize_input("se1", …)` naapurihinta-argumentista | Kausitasoitettu SE1-spot. **v2.10.0:n lisäys** — läpäisi v2.5.6:n NPK-CVaR-hedge-portin. |
| 8 | `Y_se3` | `_deseasonalize_input("se3", …)` | Kausitasoitettu SE3-spot (FennoSkan-kaapeleiden Ruotsin pää). |
| 9 | `Y_ee` | `_deseasonalize_input("ee", …)` | Kausitasoitettu EE-spot (Estlink-kaapeleiden Viron pää). |

Ridge-ennuste lasketaan kaavalla `ridge = X @ self._ridge_coef`. Kutsuja toimittaa raa'at naapurihinnat parametrillä `compute_forecast(..., recent_neighbour_prices={"se1": np.ndarray(n), "se3": …, "ee": …})`; puuttuvat tai NaN-arvot palautuvat nollasarakkeeksi kyseiselle tunnille — vastaa v2.8.x:n käyttäytymistä ilman rajasiirtohintoja.

### L3 — AR(1)-momentum

`Pipeline._ar_contribution` ([`pipeline.py:254-266`](custom_components/spot_price_predictor/pipeline.py:254)) palauttaa `φ^h · last_eta` arvoille h = 1..n. `φ` ladataan `spike_model_default.json`-tiedostosta (`ar1_phi`); `last_eta` on viimeisin havaittu AR:n jälkeinen residuaali, jota päivitetään funktiossa `Pipeline.update_with_actuals` ([`pipeline.py:423`](custom_components/spot_price_predictor/pipeline.py:423)). Kun `last_eta` on tuntematon (kylmäkäynnistys), L3 antaa nolla.

### Softplus-pohjavyöhyke ja tuntittainen biaskorjaaja

Ennen viuhkavöiden näytteistämistä keskiarvo rajataan alarajaan −5 EUR/MWh softplus-funktiolla (`price_floor.apply_floor`, oletus `_pf.DEFAULT_FLOOR_EUR_MWH`). Hitaasti liikkuva biasestimaatti vähennetään sen jälkeen `HourlyBiasCorrector`:lla (`hourly_calibration.py`, puoliintumisaika 14 vrk, lämmittely 168 tuntia). Korjattu keskiarvo on `spot_eur_mwh`-arvo jokaisella ennusterivillä.

### L4 — GPD POT -piikkimalli

`Pipeline._sample_fan_chart` (kutsutaan funktiosta `compute_forecast` kun `enable_fan_chart=True`) ottaa 500 näytettä Normaalirungon (μ, σ avaimista `stats.eta_train_mean` / `stats.eta_train_sigma`) ja Generalised Pareto -oikean- ja vasemmanpuoleisen hännän sekoituksesta (`gpd_right`, `gpd_left` -parametrilohkot: `threshold`, `shape`, `scale`, `p_exceed`). Empiiriset 5. / 25. / 50. / 75. / 95. persentiilit tunnista muodostavat `P5_eur_mwh` … `P95_eur_mwh`-avaimet jokaisella ennusterivillä.

### Pysyvät kalibroijat

Kolme kalibroijaa serialisoivat tilansa joka koordinaattorisyklillä (`Pipeline.save_state`):

| Tilatiedosto | Luokka | Konfiguroidut oletukset |
|---|---|---|
| `hourly_bias.json` | `HourlyBiasCorrector` | halflife_days = 14, warmup_hours = 168 |
| `hourly_fan_chart.json` | `HourlyFanChartCalibrator` | target_coverages = (0.5, 0.9), window = 720, min_warmup = 24 |
| `refit_monitor.json` | `RefitMonitor` | target_coverage = 0.9, drift_pp = 0.05, persistence_steps = 14 × 24 |

`RefitMonitor` nostaa `refit_recommended`-lipun kun toteutunut kattavuus poikkeaa enemmän kuin 5 prosenttiyksikköä tavoitteesta 14 peräkkäistä päivää.

## Arvioidut mutta ei tällä hetkellä käytetyt syötteet

Koordinaattori hakee useita virtoja, jotka eivät enää syötä käyttäjälle näkyvää spot-ennustetta:

- **Sahkotinin naapuri- ja historialliset hinnat, Elpriset SE1/SE3, Elering EE.** Syötetään vanhaan kestomalliin ja menneiden päivien toteutuneeseen D(k):n esimerkkilisäykseen; eivät käytössä `RIDGE_FEATURES`-piirteissä.
- **Fingrid #188** (ydinvoimatuotanto). Tällä hetkellä syötetään vanhan kestomallin `nuclear_deficit`-segmenttipiirteeseen; ei mukana `RIDGE_FEATURES`:ssa.
- **Fingrid #165 / #246 / #247** (kulutus- / tuuli- / aurinkoennusteet). Apuvirrat; eivät mukana `RIDGE_FEATURES`:ssa.
- **Nord Pool UMM** seisokkiaikataulu. Haetaan ja yhdistetään `nuclear_mw`:n kanssa tuottaen `nuclear_hourly` kestomallille; ei mukana `RIDGE_FEATURES`:ssa.

Kestoanturin kanoniset D(k)-attribuutit rakennetaan suoraan putken tuntittaisista spot-ennusteista (ks. "Kestomalli" alla), joten vanhan kestomallin segmenttikohtaiset tulosteet eivät tällä hetkellä näy käyttäjälle. Ydinvoimavajeen tai rajat ylittävien siirtokapasiteettipiirteiden uudelleen mukaan ottaminen on kokeellista työtä, joka arvioidaan erillisellä haaralla ja yhdistetään vain mitatun suorituskyvyn perusteella `main`-haaraan.

## Kestomalli — neljä D(k)-taulukkoa per päivä

Toteutus: `SpotPriceCoordinator._apply_pipeline_pre_dk` ja `_compute_duration_forecast` ([`coordinator.py`](custom_components/spot_price_predictor/coordinator.py)).

Jokaiselle paikalliselle päivälle, jossa on ennusteikkunassa 24 täyttä tuntia:

1. **Spot-puoli** — `Pipeline.compute_duration_curves` ([`pipeline.py:388-421`](custom_components/spot_price_predictor/pipeline.py:388)) lajittelee 24 tuntittaisen `mean_eur_mwh`-arvon nousevasti ja laskevasti, ja ottaa kumulatiiviset keskiarvot. Tulos: `dk_cheap_eur_mwh[24]` (i+1 halvimman tunnin keskiarvo) ja `dk_peak_eur_mwh[24]` (i+1 kalleimman tunnin keskiarvo), kummatkin monotonisia i:ssä.
2. **Kuluttajapuoli** — jokainen tuntittainen spot muunnetaan ensin kuluttajan EUR/kWh-hintaan tuntikohtaisella tariffilla (`day_rate` tunneille 07–22, `night_rate` tunneille 22–07), ja sama lajittelu + kumulatiivinen keskiarvo tuottaa `dk_cheap_eur_kwh[24]` / `dk_peak_eur_kwh[24]`.

Kaikki neljä taulukkoa ovat 0-indeksoituja; `dk_cheap_eur_mwh[3]` on päivän neljän halvimman tunnin keskiarvo (ja vastaavasti kuluttajataulukoille). Koko päivän keskiarvo palautuu indeksissä 23 kummassakin suunnassa: `dk_cheap_eur_mwh[23] == dk_peak_eur_mwh[23] == päiväkeskiarvo_spot`.

Menneille päiville `_compute_actual_duration_curves` tuottaa samat neljä taulukkoa Sahkotinin havaituista spot-hinnoista ja merkitsee päiväkortin `source: "actual"`.

## PV-tietoinen efektiivinen hinta (valinnainen)

Kun `pv_capacity_kwp > 0` (tai `pv_external_entity` on konfiguroitu), koordinaattori rikastaa jokaista ennustetuntia `effective_eur_kwh`-arvolla — yhden lisä-kWh:n joustavan kuorman marginaalikustannuksella, kun kotitalouden PV ja tyypillinen perustaso huomioidaan. Toteutus: `marginal_effective_eur_kwh` ja `net_household_cost_eur` tiedostossa [`pv_estimate.py`](custom_components/spot_price_predictor/pv_estimate.py).

Tunnille h, kun kuluttajan ostohinta on `b_h`, myyntihinta `s_h = max(0, spot − pv_sell_commission − pv_export_grid_fee)`, PV-tuotto `p_h`, ja perustaso `c_h`:

```
pv_avail_h = max(0, p_h − c_h)
from_pv    = min(1, pv_avail_h)
from_grid  = 1 − from_pv
m_h        = from_pv · s_h + from_grid · b_h
```

`m_h` on analyyttisesti rajattu välille `[s_h, b_h]`, kuvaa itsekulutushyödyn kun ylijäämä ≥ 1 kWh, ja välittää negatiiviset spot-hinnat `effective_eur_kwh`-arvoon (harvinainen mutta todellinen).

Kestoanturi laskee sen jälkeen rinnakkaiset `dk_cheap_pv_eur_kwh[24]` / `dk_peak_pv_eur_kwh[24]` -taulukot lajitelluista tuntittaisista `effective_eur_kwh`-arvoista per päivä.

### PV-syötetyypit

- **Sisäinen estimaattori (oletus)** — käyttää Open-Meteon `global_tilted_irradiance_instant` × `pv_capacity_kwp` × kallistus-/atsimuuttikorjaus × `pv_system_efficiency`. Ilmainen, 7 vrk:n horisontti.
- **Ulkoinen entiteetti (yliajo)** — aseta `pv_external_entity` mihin tahansa HA-anturiin, jonka attribuutit vastaavat jotakin tuetuista konventioista (`forecast` list-of-dict kWh:ssa, `wh_hours` dict Wh:ssä, `watts` dict W:ssä, tai `irradiance` lista automaattitunnistuksella). Forecast.Solar, EMHASS ja räätälöidyt Open-Meteo -mallit toimivat suoraan.

### Perustaso

Kaksi konfiguraatiokenttää ohjaavat PV-laskennassa käytettyä perustasoa `c_h`:

- **`annual_consumption_kwh`** (oletus 12 000) — tyypillinen KOKONAISVUOSIKULUTUS laskun mukaan. Kerrotaan 12-elementtisellä suomalaisen asuinrakennuksen kuukausittaisella kausiprofiililla (`FINLAND_RESIDENTIAL_MONTHLY_FACTORS` tiedostossa [`const.py`](custom_components/spot_price_predictor/const.py); kerroinsumma = 12,00 tarkasti, vaihteluväli ≈ ±19 %).
- **`consumption_entity`** (valinnainen) — mikä tahansa HA-kulutusanturi (kumulatiivinen kWh-laskuri, päivä/kuukausi-`utility_meter`, tai hetkellinen tehoanturi). Integraatio tunnistaa anturin tyypin, tasaa sen 14 vrk:n liukuvalla ikkunalla + 5 % hysteresis-aluekiellolla, ja välimuistittaa tuloksen kohteeseen `.storage/spot_price_predictor_consumption_cache.json` (lasketaan uudelleen enintään kerran päivässä).

### Vakausinvariantti

`_resolve_baseload(ts)` EI saa kutsua `hass.states.get`-funktiota tai mitään HA-entiteetinlukurajapintaa — varmistettu grep-testillä tiedostossa `tests/test_coordinator_pv.py`. Kun `consumption_entity = ""`, perustaso on deterministinen funktio `(annual_consumption_kwh, vuoden tunti)` -syötteistä ja ennuste on täysin avoin silmukka optimoijaa kohden. Kun `consumption_entity` on asetettu, 14 vrk:n liukuva tasaus + 5 % hysteresis pitää suljetun silmukan vahvistuksen selvästi alle yhden, joten EMHASSin päiväkohtainen uudelleenajoittaminen ei voi luoda värähtelyä.

## DtACI-kalibrointi (valinnainen)

Kun `enable_dtaci_dk = true`, koordinaattori kietoo D(k)-käyrät mukautuvilla konformaalivöillä. Toteutus: [`dk_dtaci.py`](custom_components/spot_price_predictor/dk_dtaci.py) (`DkDtACIBundle`) + [`dtaci_integration.py`](custom_components/spot_price_predictor/dtaci_integration.py).

- **48 DtACI-instanssia per alue** — yksi per `(direction, k)` suunnalle direction ∈ {cheap, peak} ja k = 1..24. Jokainen instanssi seuraa omaa residuaalijakaumaansa, alpha-arvoa, dominoivaa gammaa, painoentropiaa ja instanssikohtaista bias-EMA:a.
- **Vyöt kirjoitetaan takaisin** 24-elementtisinä taulukoina `dk_cheap_lower_eur_kwh`, `dk_cheap_upper_eur_kwh`, `dk_peak_lower_eur_kwh`, `dk_peak_upper_eur_kwh` per päivärivi kun instanssit ovat lämmittäneet. Ennen lämmittelyä vyöt romahtavat pisteen ennusteeseen (tarkoituksellinen — ei valheellista luottamusta).
- **Diagnostiikka** näkyy kestoanturissa avaimilla `dtaci_diagnostics`, `dtaci_warmup_status`, `dtaci_target_coverage`, `dtaci_fi_mean_coverage`, `dtaci_fi_mean_width_eur_kwh`, `dtaci_fi_warm_instances`, `dtaci_fi_total_instances`, `dtaci_min_n_updates`.

Tila tallennetaan tiedostoon `<config>/.storage/spot_price_predictor_dtaci_dk_fi.json`. Lämmittely kestää noin 5 päivää päivittäisistä yhteensovituksista.

Katso algoritmin yksityiskohdat (Gibbs & Candès JMLR 2024) ja vianmääritys tiedostosta [docs/dtaci_layer.md](docs/dtaci_layer.md).

## PV-tietoinen riski (v2.11.0)

Jokainen `daily_forecast[i]`-rivi kantaa — kun PV on käytössä — neljän kentän PV-tietoisen riskilohkon, jonka tuottaa päiväkohtainen CVaR-moduuli:

```
pv_aware_cvar95_eur_kwh        # efektiivisen EUR/kWh:n hännän keskiarvo
                               #   yhteisten hinta+PV-skenaarioiden yli
pv_aware_self_consumed_kwh     # odotettu paikalla käytetty PV (paikkojen keskiarvo)
pv_aware_exported_kwh          # odotettu verkkoon viety PV (paikkojen keskiarvo)
pv_aware_data_provenance       # luottamuslippu, ks. alla
```

Laskenta sijaitsee tiedostossa [`pv_aware_cvar.py`](custom_components/spot_price_predictor/pv_aware_cvar.py) ja kutsuu jaettua kustannusydintä [`pv_cost_kernel.cost_distribution()`](custom_components/spot_price_predictor/pv_cost_kernel.py). Jokaiselle ennustepäivälle:

1. Päivän 24 tuntittaista `consumer_eur_kwh`-, `sell_eur_kwh`- ja `pv_production_kwh`-arvoa luetaan tuntikohtaisilta ennusteriviltä.
2. 24 tuntittaista kulutusarvoa tulee joko ulkoisesta EMA-profiili-entiteetistä (ks. "Kulutusprofiilin lataaja" alla) tai integraation oletusperustasosta.
3. Parametrinen skenaariotuottaja ([`pv_aware_cvar._sample_pv_paths`](custom_components/spot_price_predictor/pv_aware_cvar.py)) tuottaa `N_PATHS = 200` log-normaalia perturbattua PV-polkua (keskiarvoa säilyttäviä, `REL_STD = 0.30`), kalibroituna vastaamaan empiirisen Phase-A pilvi-bootstrapin CVaR-leveyttä.
4. Kustannusydin laskee toteutuneen efektiivisen EUR/kWh:n per polku; pahimman 5 %:n hännän keskiarvosta tulee `pv_aware_cvar95_eur_kwh`.

PV-nettoutetut `dk_cheap_pv_eur_kwh[24]` / `dk_peak_pv_eur_kwh[24]` -taulukot säilyvät päiväkortilla **diagnostisina, joustava-kWh -approksimaatioina** kojelautoja varten. Kuormakohtaiset optimoijat (lämpösäätäjä, EV-lataaja, akkulaturi) **muodostavat oman per-kuorma α:nsa** käyttäen `forecast[h]`-rivien tuntikohtaisia `consumer_eur_kwh` (osto) ja `sell_eur_kwh` (myynti) -arvoja — perusteet kohteessa [studies/results/pv_adjusted_buy_sell_duration_curves.md](studies/results/pv_adjusted_buy_sell_duration_curves.md).

### Kulutusprofiilin lataaja

Moduuli: [`consumption_profile_loader.py`](custom_components/spot_price_predictor/consumption_profile_loader.py). Kaksi profiilidatan lähdettä:

1. **Ulkoinen EMA-moduuli** (esim. `HA-consumption-profiler`, erillinen repo): kun `CONF_CONSUMPTION_PROFILE_ENTITY` on asetettu kyseisen sensorin entiteetti-ID:hen, lataaja lukee sen attribuutit skeemasta [`docs/household_profile_schema.md`](docs/household_profile_schema.md): `mean_kwh_per_hour`, `shape_hour_weekday` (7×24), `monthly_factor` (12), `data_provenance`.
2. **Synteettinen varatakenttä**: jos asetus puuttuu tai on lukukelvoton, käytetään yleistä optimoimatonta suomalaisen kotitalouden profiilia (bimodaalinen tunti-of-day iltahuipulla, lämmityspainotteinen kuukausikerroin), kalibroituna asetukseen `CONF_ANNUAL_CONSUMPTION_KWH`. Provenance: `"synthetic_cold_start"`.

Synteettinen varatakenttä EI ole johdettu mistään yksittäisestä käyttäjän datasta — yksityisyyssopimus on dokumentoitu kohteessa [docs/household_profile_schema.md](docs/household_profile_schema.md) ja vahvistettu pre-commit-koukulla ([scripts/check_no_private_data.py](scripts/check_no_private_data.py)).

## Anturien attribuuttiviite

### Spot Price Forecast -anturi (Nordpool-yhteensopiva) — v2.11.0

`sensor.spot_price_forecast_fi` paljastaa putken L1+L2+L3+L4-tuotoksen Nordpool-integraatiomuodossa.

| Attribuutti | Tyyppi | Kuvaus |
|---|---|---|
| state | float (EUR/kWh) | Nykyisen tunnin spot-ennuste |
| `raw_today` | lista `{start, end, value}` | Tämänpäiväinen paikallispäiväkohtainen tuntittainen ennuste (EUR/kWh) |
| `raw_tomorrow` | lista `{start, end, value}` | Huomisen paikallispäiväkohtainen tuntittainen ennuste (EUR/kWh) |
| `raw_extended` | lista `{start, end, value}` | Koko 170-tunnin horisontti (tänään + 6 päivää). Integraation ainutlaatuinen lisäarvo. |
| `today_min` / `today_avg` / `today_max` | float (EUR/kWh) | `raw_today`-arvojen tilastot |
| `tomorrow_min` / `tomorrow_avg` / `tomorrow_max` | float (EUR/kWh) | Samat huomenna |
| `forecast_horizon_h` | int | `raw_extended`-pituus |
| `currency`, `unit`, `source` | string | `"EUR"`, `"kWh"`, `"spot_price_predictor L1+L2+L3+L4"` |
| `confidence_band` | dict `{p5: [...], p95: [...]}` | L4-viuhkavyöt per tunti (EUR/kWh). Valinnainen. |
| `last_updated` | ISO-aikaleima | Viimeinen koordinaattorisykli |

Empiirinen tarkkuus (12 kuukauden held-out -tausta-ajo välimuistissa olevista hinnoista + säästä, ks. [studies/results/exp_spot_price_forecast_accuracy.md](studies/results/exp_spot_price_forecast_accuracy.md)):

- **Kylmäkäynnistyksen alaraja** (tuore asennus, ei kalibroijan historiaa): MAE 22,5 EUR/MWh keskimääräisellä toteutuneella hinnalla 51,8 EUR/MWh; R² +0,71; 50 %:n vyökate 49 % (tavoite 50 %); 90 %:n vyökate 74 % (alimitoitettu kylmäkäynnistyksessä).
- **Lämmin tila** noin 30–60 päivän jälkeen (kalibroijat lämmenneet): MAE ≈ 10 EUR/MWh, R² ≈ 0,91, 90 %:n vyökate ≈ 92 %. Numerot v2.10.1-julkaisun tausta-ajosta saman datan koulutus/testijaolla.

### Price Forecast -sensori — forecast-rivin avaimet

Jokainen rivi `forecast[]`-taulukossa (pituus 170):

| Avain | Tyyppi | Lähde |
|---|---|---|
| `timestamp` | string (ISO 8601 UTC) | tuntiindeksi koordinaattorista |
| `spot_eur_mwh` | float | `Pipeline.compute_forecast`-keskiarvo (pohjavyöhyke + biaskorjaus sovellettuna) |
| `consumer_eur_kwh` | float | `spot_eur_mwh / 1000 + seller_margin + transfer + energy_tax) × VAT`, tuntikohtaisella päivä-/yötariffilla |
| `wind` | float (m/s) | Open-Meteo kapasiteettipainotettu 120 m tuuli |
| `solar` | float (W/m²) | Open-Meteo GHI kapasiteettipainotettu |
| `temp` | float (°C) | Open-Meteo lämpötila kapasiteettipainotettu |
| `P5_eur_mwh`, `P25_eur_mwh`, `P50_eur_mwh`, `P75_eur_mwh`, `P95_eur_mwh` | float | L4 GPD POT viuhkapersentiilit |
| `pv_production_kwh`, `baseload_kwh`, `effective_eur_kwh`, `net_household_cost_eur`, `is_export_hour`, `sell_eur_kwh` | vaihtelee | Vain kun PV on käytössä |

### Duration Forecast -sensori — päiväkortin avaimet

Jokainen rivi `daily_forecast[]`-taulukossa (enintään 7):

| Avain | Tyyppi | Huomautukset |
|---|---|---|
| `date`, `weekday`, `source` | string | `source ∈ {"forecast", "actual"}` |
| `dk_cheap_eur_mwh`, `dk_peak_eur_mwh` | float[24] | Spot EUR/MWh, 0-indeksoituna, monotoninen i:ssä |
| `dk_cheap_eur_kwh`, `dk_peak_eur_kwh` | float[24] | Kuluttaja EUR/kWh, tuntikohtainen tariffi sovellettu |
| `dk_cheap_pv_eur_kwh`, `dk_peak_pv_eur_kwh` | float[24] | PV-tietoiset versiot (yhden perustason "joustava kWh" -approksimaatio). Vain kun PV on käytössä. Vain kojelaudoille; kuormakohtaiset optimoijat muodostavat oman α:nsa tuntikohtaisesta osto/myynti-datasta. |
| `pv_aware_cvar95_eur_kwh` | float | **v2.11.0.** Pahimman 5 %:n yhdistettyjen hinta+PV-skenaarioiden efektiivisen kustannuksen hännän keskiarvo. EUR/kWh. Vain kun PV on käytössä. |
| `pv_aware_self_consumed_kwh`, `pv_aware_exported_kwh` | float | **v2.11.0.** Odotettu PV paikallisesti käytetty / verkkoon viety tänä päivänä, skenaarioiden keskiarvo. Vain kun PV on käytössä. |
| `pv_aware_data_provenance` | string | **v2.11.0.** `"synthetic_cold_start"` / `"ema_blended"` / `"ema_warm"` / `"coordinator_baseload"` — luottamuslippu CVaR:n taustalla olevalle kulutusprofiilille. |
| `dk_cheap_lower_eur_kwh`, `dk_cheap_upper_eur_kwh`, `dk_peak_lower_eur_kwh`, `dk_peak_upper_eur_kwh` | float[24] | DtACI-vyöt. Vain kun DtACI on käytössä ja instanssit ovat lämmittäneet. |

### Diagnostiikka koordinaattorin tuloksessa

| Avain | Lähde |
|---|---|
| `pipeline_diagnostics` | `pipeline_bias_eur_mwh`, `pipeline_ar1_phi`, `pipeline_n_features`, `pipeline_floor_eur_mwh` |
| `dtaci_diagnostics` | `DkDtACIBundle.diagnostics()`-palaute kun DtACI on käytössä |
| `data_sources_active` | Tekstitiivistelmä haetuista virroista |
| `last_update`, `stale`, `data_age_minutes` | Standardi tilatieto |
| `pv_enabled`, `pv_capacity_kwp`, `pv_source`, `baseload_kwh_per_hour`, `current_effective_eur_kwh` | PV-metadata (aina lähetetty) |

## Home Assistant -palvelut

Rekisteröity tiedostossa [`__init__.py`](custom_components/spot_price_predictor/__init__.py); skeemat tiedostossa [`services.yaml`](custom_components/spot_price_predictor/services.yaml):

| Palvelu | Argumentit | Vaikutus |
|---|---|---|
| `spot_price_predictor.retrain_models` | `layers` (valinnainen lista, osajoukko joukosta `{seasonal, spike, solar}`); `fingrid_api_key` (valinnainen, luettavissa myös ympäristömuuttujasta) | Sovittaa luetellut (tai kaikki) artefaktit uudelleen atomisesti, lataa `Pipeline`:n uudelleen jokaisella aktiivisella koordinaattorilla, laukaisee `spot_price_predictor_models_retrained`-tapahtuman valmistuessa. |
| `spot_price_predictor.force_refresh` | ei mitään | Käynnistää välittömän koordinaattorin päivityksen. |
| `spot_price_predictor.model_info` | ei mitään | Julkaisee pysyvän ilmoituksen artefaktien metadatasta. |
| `spot_price_predictor.upload_coefficients` | `file_path` (valinnainen) tai `json_data` (valinnainen) | Korvaa vanhan v2.2 käyttäjäkerrointiedoston. |
| `spot_price_predictor.reset_coefficients` | ei mitään | Palauttaa mukana toimitetut oletuskertoimet. |

Integraatio julkaisee tapahtuman `spot_price_predictor_models_retrained` onnistuneen uudelleenkoulutuksen jälkeen, payload `{"result": …, "reloaded_coordinators": …}`.

## Päivitystiheys

Määritelty tiedostossa [`const.py:155-161`](custom_components/spot_price_predictor/const.py:155):

| Vakio | Arvo | Käyttö |
|---|---|---|
| `UPDATE_INTERVAL_WEATHER` | 21 600 s (6 h) | Koordinaattorin jaksottainen päivitys onnistumisen jälkeen. |
| `UPDATE_INTERVAL_FINGRID` | 3 600 s (1 h) | Fingrid-aineistojen päivitysväli. |
| `FORECAST_HOURS` | 170 | Tuntittaisen ennustetaulukon pituus. |

## Alueellinen lokalisointi

Järjestelmää ohjaa `config/regions/finland.yaml`. Uuden alueen tukeminen:

1. Tunnista 5–8 säämittauspaikkaa, painotettuna asennetun tuuli- / aurinkokapasiteetin ja väestön mukaan.
2. Luo uusi YAML-tiedosto (esim. `sweden.yaml`), jossa määritellään paikallinen hinta-API, pyhäpäiväsäännöt, kuluttajahinnoittelu ja (valinnaisesti) naapurivyöhykkeiden hintalähteet.
3. Sovita uudelleen `spot_price_predictor.retrain_models`-palvelulla alueellisen välimuistissa olevan datan kanssa.

Putkiarkkitehtuuri (L1 kausi + L2 fysiikkapiirteinen Ridge + L3 AR(1) + L4 GPD POT) soveltuu mille tahansa sähkömarkkinalle, jolla on säävetoista tuotantoa.

## Projektin rakenne

```
HA-spot-price-predictor/
├── README.md
├── TECHNICAL_GUIDE.md
├── TEKNINEN_TOTEUTUS.md
├── INSTALLATION.md
├── config/regions/finland.yaml
├── custom_components/spot_price_predictor/
│   ├── __init__.py                 # sisääntulo + palvelujen rekisteröinti
│   ├── coordinator.py              # DataUpdateCoordinator + putken orkestrointi
│   ├── pipeline.py                 # Pipeline (L1+L2+L3+L4+pohjavyöhyke+bias-EMA)
│   ├── seasonal_decomposition.py   # L1-komponenttien sovitus / haku
│   ├── hourly_calibration.py       # HourlyBiasCorrector / HourlyFanChartCalibrator / RefitMonitor
│   ├── dk_dtaci.py                 # DkDtACIBundle (48 instanssia per alue)
│   ├── dtaci_integration.py        # paketti ↔ duration_forecast-johdotus
│   ├── price_floor.py              # softplus-pohjavyöhyke −5 EUR/MWh:llä
│   ├── solar_clear_sky.py          # selkeän taivaan × pilvisyys -aurinkomalli
│   ├── pv_estimate.py              # sisäinen PV-estimaattori + marginaalinen efektiivinen hinta
│   ├── pv_cost_kernel.py           # jaettu yhteiskustannus + CVaR -kirjasto
│   ├── pv_aware_cvar.py            # päiväkohtainen PV-tietoinen CVaR (parametrinen skenaario)
│   ├── consumption_profile_loader.py  # EMA-profiililukija + synteettinen varatakenttä
│   ├── retrain.py                  # retrain_models-orkestraattori
│   ├── sensor.py                   # sensorientiteetit
│   ├── api_client.py               # asynkroniset API-asiakkaat
│   ├── const.py                    # vakiot, operaattorit, konfiguraatioavaimet
│   ├── config_flow.py              # HA:n asennusvelho
│   └── data/
│       ├── seasonal_components_default.json
│       ├── spike_model_default.json
│       ├── solar_submodel_default.json
│       └── finland.yaml
├── studies/                        # uudelleenkoulutus-skriptit + historialliset analyysit
└── tests/                          # pytest-paketti (471 läpäisty v2.11.0:ssa)
```
