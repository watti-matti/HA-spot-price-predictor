# Tekninen toteutus: HA-spot-price-predictor (v2.8.0)

Sähkön kuluttajahinnan ja D(k) = CVaR -kestokustannusten ennustaminen Home Assistantiin. Tuottaa 170 tunnin kuluttajahintaennusteen (EUR/kWh) ja 7 vrk D(k) halpa/kallis -kestokäyrät kuormanohjauksen kustannusoptimointiin, käyttäen nelitasoista putkea — L1 kausivaihteludekompositio, L2 fysiikkapohjainen Ridge, L3 AR(1)-momentum, L4 GPD POT -piikkimalli — yhdessä softplus-pohjavyöhykkeen ja tuntittaisen DtACI-kalibroijan kanssa. Valinnaisesti rikastaa jokaisen ennustetunnin PV-tietoisella marginaalisella efektiivisellä hinnalla `m_h` ja PV-tietoisilla D(k)-käyrillä, kun käyttäjä konfiguroi kotitalouden aurinkopaneelit.

## Arkkitehtuuri

Järjestelmässä on kaksi vaihetta: **päättely** (Home Assistant -integraatio, jatkuvasti päällä) ja **uudelleenkoulutus** (sovittaa mukana tulevat artefaktit pyynnöstä uudelleen Home Assistantin palvelun kautta). Molemmat vaiheet jakavat saman koodipolun `custom_components/spot_price_predictor/`:in alla.

### Uudelleenkoulutusvirta (Home Assistantin palvelu `spot_price_predictor.retrain_models`)

```
Välimuistissa olevat hinta-     ──> studies/build_seasonal_components.py  ──> data/seasonal_components_default.json
ja sääparquetit / Sahkotin          studies/v2513_layer4_spike_model.py     ──> data/spike_model_default.json
                                    studies/v253_solar_submodel.py          ──> data/solar_submodel_default.json
                                                                                  │
                                                                                  v
                                                                Atomiset JSON-kirjoitukset + V26Pipeline uudelleenlataus
                                                                (laukaisee tapahtuman spot_price_predictor_models_retrained)
```

### Home Assistant -käyttöönotto

```
Open-Meteo  ──┐
Elpriset    ──┼──> Koordinaattori ──> V26-putki (L1 kausi + L2 Ridge + L3 AR(1) + L4 GPD POT)
Elering     ──┤    + Datahaku        + Softplus-pohjavyöhyke + Tuntittainen DtACI-kalibroija
Fingrid     ──┤    + Tariffi-        + Kestomalli (segmenttihierarkkinen Ridge + PAVA)
Sahkotin    ──┤    muunnos                      │
Nord Pool UMM ┘                                 v
                                       Spot-/kuluttajahinta-ennuste (170h, P5/P25/P50/P75/P95 -viuhka)
                                       + D(k) halpa/kallis (7 vrk)
                                       + (valinnainen) PV-tietoinen efektiivinen hinta ja PV-D(k)
                                       ↑
                                       └── Open-Meteo irradianssi (sisäinen)
                                           TAI pv_external_entity (Forecast.Solar / EMHASS / malli)
```

### Kojelaudat

Kaksi visualisointikojelautaa:

| Kojelauta | Skripti | Tarkoitus |
|-----------|---------|-----------|
| `model_dashboard.html` | `model_dashboard.py` | Mallin seuranta: D(k) tarkkuus, piirteiden tärkeys, liukuva Spearman, lambda-pyyhkäisy |
| `forecast.html` | `forecast_dashboard.py` | Reaaliaikainen 7 vrk ennuste: D(k) kestokäyrät, tuntihinnat, sääkonteksti |

---

## Datalähteet

Kaikki datalähteet konfiguroidaan tiedostossa `config/regions/finland.yaml`.

### Pakolliset (ilmaiset, ei tunnistautumista)

| Lähde | Tarkoitus | Pyyntöraja |
|-------|-----------|------------|
| [Sahkotin API](https://sahkotin.fi/prices) | Suomen Nord Pool spot-hinnat (EUR/MWh) | Rajoittamaton |
| [Open-Meteo API](https://api.open-meteo.com) | Tuuli (120m), aurinko (45° kallistus), lämpötila | 10 000/vrk |
| [Open-Meteo Historical Forecast](https://historical-forecast-api.open-meteo.com) | Historiallinen säädata koulutukseen | 10 000/vrk |

### Rajat ylittävät hintalähteet (ilmaiset, ei tunnistautumista)

| Lähde | Haetut alueet | Käyttääkö nykyinen ei-kausi-osan malli? |
|-------|---------------|-----------------------------------------|
| [Elpriset](https://www.elprisetjustnu.se/api/v1/prices) | SE3 | Ei — käytössä kestomallissa ja kojelaudoissa |
| [Elpriset](https://www.elprisetjustnu.se/api/v1/prices) | SE1 | Ei — apukäyttö (spread / historiakonteksti) |
| [Elering API](https://dashboard.elering.ee/api/nps/price) | EE | Ei — käytössä kestomallissa ja kojelaudoissa |

Rajat ylittävät hinnat ovat osa datakerrosta kestomallia ja historiakonteksti-attribuutteja varten, mutta ne eivät ole ei-kausi-osan hintamallin piirteitä.

### Valinnainen verkkodata (ilmainen API-avain)

| Lähde | Tarkoitus |
|-------|-----------|
| [Fingrid Open Data](https://data.fingrid.fi) | Ydinvoimatuotanto (#188), kulutus-/tuuli-/aurinkoennusteet (#165 / #246 / #247) — kestomalli ja apuvirrat |

Rekisteröidy ilmaiseksi osoitteessa data.fingrid.fi. Ilman Fingrid-avainta ei-kausi-osan hintamalli toimii muuttumattomana; kestomalli käyttää säävetoisia ominaisuuksiaan.

### Ydinvoimaseisokkiaikataulu (ilmainen, ei avainta)

| Lähde | Tarkoitus |
|-------|-----------|
| [Nord Pool UMM](https://ummapi.nordpoolgroup.com/messages) | Suunnitellut ydinvoimaseisokit — apuvirta, ei ei-kausi-osan hintamallissa |

---

## Piirteiden suunnittelu

Ei-kausi-osan hintamalli on tarkoituksellisesti kompakti: kuusi Ridge-piirrettä yhdistettynä AR(1)-momentumtermiin ja raskasarvoiseen piikkikerrokseen. Täydellinen piirrejärjestys on määritelty tiedostossa `custom_components/spot_price_predictor/v26_pipeline.py:68-75` (vakio `V26_FEATURES`) — alla oleva dokumentaatiotaulukko vastaa täsmälleen tätä järjestystä.

### L1 — Kausivaihteludekompositio

`custom_components/spot_price_predictor/seasonal_decomposition.py` toteuttaa additiivisen Moazeni–Powell-dekomposition: jokainen syöteaikasarja jaetaan tuntittaisiin, päivittäisiin ja viikoittaisiin komponentteihin, sovitetaan koulutushistorialla ja toimitetaan tiedostossa `data/seasonal_components_default.json`. Päättelyvaiheessa putki vähentää komponentit tuottaen kausitasoittuneet residuaalit (`Y_*`). FI-hinta käyttää koko tunti+päivä+viikko-syvyyden; lämpötila käyttää tunti+viikko; tuuli ja aurinko keskitetään paikallisesti L1:n sijaan ennen Ridge-vaihetta.

### L2 — Ei-kausi-osan Ridge-regressio

Kuusi piirrettä kasataan suunnittelumatriisiin tässä järjestyksessä (vastaa `V26_FEATURES`):

| # | Piirre (koodinimi) | Rakennetaan kohdassa | Määritelmä |
|---|---|---|---|
| 1 | `intercept` | `v26_pipeline.py:241` | Vakio 1. Kaappaa kausitasoittuneen keskiarvon. |
| 2 | `Y_fi_lag168` | `v26_pipeline.py:242` | Kausitasoittunut FI-residuaali 7 päivää aiemmin — paikallisen markkinaregiimin omanvältin muisti. Kylmäkäynnistyksessä nollia. |
| 3 | `is_workday` | `v26_pipeline.py:243`, lasketaan `:251-256` | `weekday < 5`. Kaappaa teollisuuden kysyntäkuvion. |
| 4 | `Y_sigmoid_wind_rho` | `v26_pipeline.py:244`, apufunktio `_sigmoid_turbine_rho` `:87-93` | `σ((tuuli − 7,5) / 1,5) × ρ(T) / 1,225`. Sigmoidaalinen tuuliturbiinikäyrä skaalattuna ilman suhteellisella tiheydellä; fysiikkapohjainen tarjonta-ajuri. Keskitetään paikallisesti ennen Ridgeä. |
| 5 | `Y_solar_effective` | `v26_pipeline.py:245`, apufunktio `_solar_effective` `:96-102` | `GHI × (1 − 0,004 · max(0, T_cell − 25))`, missä `T_cell = T + 0,03 · GHI`. Lämpötilakompensoitu efektiivinen irradianssi. Keskitetään paikallisesti ennen Ridgeä. |
| 6 | `Y_temp` | `v26_pipeline.py:246`, kausitasoitetaan `:239` | Kausitasoittunut lämpötila — lämmityskuorman residuaalisignaali. |

Ridge-kerroinvektori on tiedostossa `data/spike_model_default.json` avaimella `ridge_coef`; se sovelletaan kohdassa `v26_pipeline.py:367` lausekkeena `ridge = X @ self._ridge_coef`.

### L3 — AR(1)-momentum

Ennustehorisontilla `h` AR(1)-termin osuus on `φ^h · η(t₀−1)`, missä `η(t₀−1)` on viimeisin havaittu kausitasoittunut FI-residuaali ja `φ` on AR(1)-kerroin (tyypillisesti `φ ≈ 0,904`) ladattuna tiedostosta `spike_model_default.json`. Toteutus: `v26_pipeline.py:260-266`; residuaalitila päivitetään funktiossa `update_with_actuals()` (`:429-446`).

### L4 — GPD POT -piikkimalli

Normaalirungon ja yleistetyn Pareto-hännän sekoitusta otetaan 500 näytettä L1+L2+L3-pisteen ennusteen ympärillä, mikä tuottaa P5/P25/P50/P75/P95-viuhkavyöt. Parametrit tiedostossa `spike_model_default.json`:

| Parametri | Tehtävä |
|---|---|
| `stats.eta_train_mean`, `stats.eta_train_sigma` | Normaalirungon keskiarvo ja skaala residuaalille `η` |
| `gpd_right.{threshold, shape, scale, p_exceed}` | Oikean hännän ylitysmalli |
| `gpd_left.{threshold, shape, scale, p_exceed}` | Vasemman hännän ylitysmalli |

Näytteistäjän toteutus: `_sample_fan_chart` (`v26_pipeline.py:270-320`). Viuhkavyöt näkyvät `forecast`-sensorissa avaimilla `P5_eur_mwh` … `P95_eur_mwh`.

### Softplus-pohjavyöhyke ja tuntittaiset DtACI-kalibroijat

Ennen viuhkavöiden näytteistämistä L1+L2+L3-keskiarvo rajataan alarajaan −5 EUR/MWh softplus-funktiolla (`price_floor.py`). Tuntittainen DtACI-biaskorjaaja (`hourly_calibration.HourlyBiasCorrector`, puoliintumisaika 14 vrk, 168 tunnin lämmittely) vähentää hitaasti liikkuvan systemaattisen biaksen; rinnakkainen `HourlyFanChartCalibrator` mukauttaa tuntittaiset viuhkaleveydet seuraamaan 0,5:n ja 0,9:n marginaalikattavuustavoitteita. `RefitMonitor` merkitsee jatkuvan ryöminnän 14 vrk ikkunassa (`spot_price_predictor_v26/refit_monitor.json`).

### Arvioidut mutta ei tällä hetkellä aktiivisesti käytetyt syötteet

Aikaisemmassa toteutettavuustyössä tutkittiin ydinvoimavajesignaalin `nuclear_deficit ∈ [0, 1]` (Fingrid-tietojoukko #188) ja SE3:n rajat ylittävän siirtokapasiteetin / vientispredin proksin käyttöä piirteinä. Nykyinen ei-kausi-osan malli ei käytä kumpaakaan:

- Fingrid-ydinvoimadata haetaan edelleen (`coordinator.py:871`) ja näkyy kestomallissa sekä diagnostiikassa, mutta ei ole `V26_FEATURES`-listalla.
- SE3 / SE1 / EE -hinnat haetaan edelleen (`coordinator.py:852`) ja syötetään kestomalliin; v26-kutsupiste (`coordinator.py:1210-1215`) välittää vain sään ja lag168-residuaalin.

Kummankin signaalin uudelleenkäyttöönotto vaatisi tuoreen ablaation nykyistä kerroinvektoria vastaan ja on sijoitettu erilliseen kokeelliseen haaraan — katso "Avoin kysymys — syötekohtainen tarkkuusvaikutus" alla.

### Avoin kysymys — syötekohtainen tarkkuusvaikutus

Ainoa tässä repossa nykyisin julkaistu syötekohtainen ablaatio on [studies/results/v2511_physics_features.md](studies/results/v2511_physics_features.md), joka arvioi sigmoid-tuuli / aurinko / raakatuuli -variantteja perusasetelmaa vastaan. Se on hyödyllinen taustaluku fysiikkapiirteiden suunnittelulle, mutta ei pinnoittele kunkin L2-piirteen marginaalivaikutusta nykyiseen kerroinvektoriin. Leave-one-out -tutkimus nykyiselle `ridge_coef`-vektorille on ehto syötelistan muutoksille.

---

## Malliarkkitehtuuri

### Tuntimalli: Nelitasoinen putki

Tuntittaisen pisteen ennuste on kolmen additiivisen kontribuution summa, softplus-rajattu ja bias-korjattu:

```
v26_mean(h) = L1_seasonal_fi(h)
            + L2_ridge(h)          # kuusi yllä olevaa piirrettä
            + L3_ar(h)              # φ^h · η(t₀−1)
            - hourly_bias_ema(h)    # DtACI-biaskorjaaja
            (softplus-pohjavyöhyke −5 EUR/MWh:llä)
```

Täysi näytteistäjä tuottaa sen jälkeen viuhkavyöt `P5_eur_mwh` … `P95_eur_mwh` pisteen ennusteen (`spot_eur_mwh`) ympärille. Julkinen sisääntulo: `V26Pipeline.compute_forecast` (`v26_pipeline.py:324`).

Rakentamisen yhteydessä ladattavat artefaktit:

- `data/seasonal_components_default.json` — L1-komponentit FI-hinnalle ja sääsyötteille.
- `data/spike_model_default.json` — L2 Ridge-kertoimet (`ridge_coef`), L3 AR(1) (`ar1_phi`), L4 Normaalirungon + GPD-hännän parametrit (`stats`, `gpd_left`, `gpd_right`).
- `data/solar_submodel_default.json` — selkeän taivaan × pilvisyys -aurinkotuotantomalli, jota PV-tietoinen polku käyttää.

Pysyvä kalibroijan tila kansiossa `<config>/.storage/spot_price_predictor_v26/`:

- `hourly_bias.json` — DtACI-biaskorjaajan tila.
- `hourly_fan_chart.json` — viuhkan DtACI-paketti kattavuustavoitteittain.
- `refit_monitor.json` — refit-monitorin ryömintälaukaisutila.

### Kestomalli: Segmenttihierarkkinen Ridge + PAVA

Ennustaa D(k) = keskimääräisen spot-hinnan halvimmille k tunnille päivässä. D(k) on matemaattisesti ekvivalentti ehdollisen riskin arvon (CVaR, Conditional Value-at-Risk) kanssa päivänsisäisestä hintajakaumasta tasolla α = k/24, mikä tekee siitä luonnollisen kustannusmittarin kuormien ajoitukseen: "ajoita halvimmille k tunnille" minimoi CVaR:n.

**PAVA** (Pool Adjacent Violators Algorithm) on isotonisen regression menetelmä, joka pakottaa monotonisuuden. Koska D(k) on määritelmän mukaan ei-vähenevä — useampien tuntien lisääminen keskiarvoon voi sisältää vain yhtä kalliita tai kalliimpia tunteja — PAVA yhdistää itsenäisten Ridge-ennusteiden rikkomukset keskiarvoistamalla vierekkäisiä pareja kunnes D(1) ≤ D(2) ≤ ... ≤ D(N) toteutuu kaikkialla.

**Arkkitehtuuri (halpa/kallis -kaksisuuntainen koulutus):**
- 4 päiväsegmenttiä tariffirajojen mukaan: yö (22-07, 9 tasoa), aamu (07-12, 5 tasoa), keskipäivä (12-18, 6 tasoa), ilta (18-22, 4 tasoa). Yhteensä 24 tuntipaikkaa.
- Jokainen `(segmentti, suunta, k)`: itsenäinen Ridge-malli. Jokainen segmentti sisältää `cheap_models` (k = 1..n_levels) ja `peak_models` (k = 1..n_levels). Mukana tulevia Ridge-sovituksia yhteensä = 2 × (9 + 5 + 6 + 4) = **48 pientä mallia**.
- Segmenttikohtaiset **12 piirrettä** (segmentin tuntien yli laskettuja keskiarvoja):
  `wind_mean`, `solar_mean`, `hdd_mean`, `se3_mean`, `se1_mean`, `nuclear_deficit`, `is_workday`, `month_sin`, `month_cos`, `wind_log_scarcity`, `net_load_mean`, `net_load_squared_mean`. Kaksi viimeistä asetetaan nollaksi kun Fingrid-kuormitusennusteita ei ole saatavilla, vastaten koulutuspuolen oletusarvoa.
- Log-lineaarinen kohde: `log(D(k) + 100)`
- Unohtamiskerroin λ = 0,960 (puoliintumisaika 17 päivää, optimoitu pyyhkäisyllä)
- PAVA isotoninen jälkikäsittely suunnittain:
  - Halpa pää: pakottaa `dk_cheap[0] ≤ dk_cheap[1] ≤ … ≤ dk_cheap[11]`
  - Kallis pää: pakottaa `dk_peak[0] ≥ dk_peak[1] ≥ … ≥ dk_peak[11]`
- Segmentistä päivään -rekonstruktio: jokainen segmentti tuottaa oman lajitellun hintavektorinsa; segmentit yhdistetään 24 tunnittaiseksi ennusteeksi; `compute_dk_cheap_peak()` tuottaa kaksi 12-elementtistä taulukkoa, jotka näkyvät sensorissa.

**Suorituskyky (Spearmanin järjestyskorrelaatio, viimeiset 365 päivää):**

| Kestotaso | Käyttötapaus | ρ |
|:-:|:-:|:-:|
| D(1) | Halvin tunti | 0,898 |
| D(4) | Halvimmat 4h | 0,930 |
| D(8) | Halvimmat 8h | 0,937 |
| D(24) | Päivän keskiarvo | 0,940 |

**Tulosteartefaktit:** nelitasoisen tuntimallin parametrit sijaitsevat tiedostoissa `data/spike_model_default.json` ja `data/seasonal_components_default.json`; kestomalli sisältää AR(2)-parametrit `ar_se3`:lle ja `ar_ee`:lle sekä kaksisuuntaiset halpa/kallis-kertoimet (48 segment-suunta-k Ridge-sovitusta).

---

## Konfiguraatio

Kaikki säädettävät parametrit on keskitetty tiedostoon `config/regions/finland.yaml`.

| Osio | Parametrit |
|------|-----------|
| `region` | Nimi, aikavyöhyke, leveysaste, valuutta, tarjousalue |
| `price_source` | Sahkotin API, yksikkömuunnos |
| `weather_source` | Open-Meteo URL:t, 7 sijainnin määrittelyt kapasiteettipainoilla |
| `neighbor_price_sources` | Elpriset (SE1, SE3), Elering (EE) rajapinnat |
| `grid_sources` | Fingridin ydinvoimatietojoukko |
| `demand` | HDD-kynnys, huipputunnit, saunatunnit, tuulen nimellisarvo |
| `holidays` | Kiinteät, pääsiäispohjaiset ja erikoissääntöpyhäpäivät |
| `consumer_pricing` | ALV, energiavero, myyjän marginaali, operaattoritariffit |
| `features` | Tuulikynnykset, AR-normalisointi, AR-vakausrajat |
| `training` | Vuodet, testijako, puoliintumisaika, Ridge alpha, tehovenytysrajat |
| `duration_model` | Lambda, segmentit, piirteet, log-offset, exp-katto |

---

## Kuluttajahinnan laskenta

**Kaava:** `(max(0, spot_EUR_MWh) / 1000 + marginaali + siirtohinta + energiavero) × ALV` [EUR/kWh]

Konfiguroitavissa operaattorikohtaisesti tiedostossa `finland.yaml`. Oletus: Elenia (päivä 3,61, yö 2,20 c/kWh), ALV 25,5%, energiavero 2,325 c/kWh, myyjän marginaali 0,00 c/kWh (aseta sähkösopimuksesi mukaan).

### Ennustesensorit (luodaan aina)

| Sensori | Tila | Yksikkö | Kuvaus |
|---------|------|---------|--------|
| Price Forecast | Nykyinen kuluttajahinta | EUR/kWh | 170h tuntiennuste: spot, kuluttajahinta, sää per tunti |
| Duration Forecast | Tämän päivän D(4) | EUR/kWh | 7 vrk D(k) halpa/kallis -kestokäyrät (12 tasoa kumpaankin suuntaan), valinnaisesti PV-tietoiset |

#### Price Forecast -attribuutit

| Attribuutti | Tyyppi | Kuvaus |
|-------------|--------|--------|
| `forecast` | array[170] | Per tunti: `{timestamp, spot_eur_mwh, consumer_eur_kwh, wind, solar, temp, P5_eur_mwh, P25_eur_mwh, P50_eur_mwh, P75_eur_mwh, P95_eur_mwh}`. `spot_eur_mwh` on putken pisteen ennuste (EUR/MWh); `P*_eur_mwh` ovat L4 GPD POT -kerroksen viuhkapersentiilit. |
| `current_spot_eur_mwh` | float | Nykyisen tunnin spot-hinta (EUR/MWh) |
| `week_min_eur_kwh` | float | Ennusteikkunan minimi kuluttajahinta |
| `week_avg_eur_kwh` | float | Ennusteikkunan keskimääräinen kuluttajahinta |
| `week_max_eur_kwh` | float | Ennusteikkunan maksimi kuluttajahinta |
| `operator` | string | Konfiguroitu jakeluverkko-operaattori |
| `last_update` | datetime | Viimeisin onnistunut datapäivitys |
| `data_sources_active` | string | Aktiiviset datalähteet |
| `stale` | bool | True jos data on vanhempaa kuin kynnysarvo |
| `data_age_minutes` | int | Minuutteja viimeisimmästä onnistuneesta hausta |

#### Duration Forecast -attribuutit — D(k) halpa/kallis -käyrät

`daily_forecast`-attribuutti sisältää enintään 7 päivää. Jokaiselle päivälle annetaan neljä 24-paikkaista taulukkoa (0-indeksoituna): halpa- ja kallispään käyrät sekä spot-hinnassa (EUR/MWh) että kuluttajahinnassa (EUR/kWh). Anturin tila on tämän päivän `dk_cheap_eur_kwh[3]` — halvimpien 4 tunnin keskihinta.

| Attribuutti | Muoto | Yksikkö | Kuvaus |
|-------------|-------|---------|--------|
| `daily_forecast` | array[≤7] | — | Yksi rivi per päivä |
| `daily_forecast[i].date` | string | — | ISO-päivämäärä (VVVV-KK-PP) |
| `daily_forecast[i].weekday` | string | — | Mon–Sun |
| `daily_forecast[i].source` | string | — | `forecast` tai `actual` (toteutuneet päivät Sahkotinista) |
| `daily_forecast[i].dk_cheap_eur_mwh` | float[24] | EUR/MWh | Päivän (i+1) halvimman tunnin keskimääräinen spot-hinta, i = 0..23 (monotonisesti ei-laskeva) |
| `daily_forecast[i].dk_peak_eur_mwh`  | float[24] | EUR/MWh | Päivän (i+1) kalleimman tunnin keskimääräinen spot-hinta, i = 0..23 (monotonisesti ei-nouseva) |
| `daily_forecast[i].dk_cheap_eur_kwh` | float[24] | EUR/kWh | Sama halpa-pään käyrä kuluttajahintana (tuntikohtainen tariffimuunnos sovellettu) |
| `daily_forecast[i].dk_peak_eur_kwh`  | float[24] | EUR/kWh | Sama kallis-pään käyrä kuluttajahintana |
| `forecast_days` | int | — | Päivien lukumäärä (enintään 7) |

**Käyttötavat:**
- Halvimmat k tuntia päivänä d: `daily_forecast[d].dk_cheap_eur_kwh[k-1]` arvoille k = 1..24 (siirrettävien kuormien aikataulutus)
- Kalleimmat k tuntia päivänä d: `daily_forecast[d].dk_peak_eur_kwh[k-1]` arvoille k = 1..24 (varakapasiteetti, pahimman tapauksen suunnittelu)
- Yhtälöllinen tarkistus: `cheap[11] + peak[11] = 2 × päiväkeskiarvo` (pätee aina numeerisen kohinan tarkkuudella)
- Kuluttajahinnat sisältävät segmenttikohtaisen tariffimuunnoksen: yötunnit yösiirtotariffilla, päivätunnit päivätariffilla, yhdistetty ja uudelleenlajiteltu

Migraatio-ohje kolmansille osapuolille: katso [docs/dk_cheap_peak_migration.md](docs/dk_cheap_peak_migration.md).

### Todellinen hinta -sensorit (valinnainen, Nordpool)

| Sensori | Yksikkö | Kuvaus |
|---------|---------|--------|
| Spot Electricity Price | EUR/kWh | Todellinen kuluttajahinta Nordpoolista jatkuvalla aikajanalla |
| Spot Electricity Selling Price | EUR/kWh | Spot miinus aurinkosähkön myyntipalkkio |

### Suunnitteluperiaate

Tämä integraatio tuottaa **ainoastaan ennusteita**. D(k) = CVaR -kestomatriisi on ensisijainen rajapinta alavirtajärjestelmille — lämpöoptimointi ja kuormanohjaus käyttävät D(k)-arvoja vastatakseen kysymykseen "paljonko k tunnin käyttö maksaa tänään per kWh?" Tämä selkeä erottelu mahdollistaa kummankin komponentin itsenäisen korvaamisen.

**Price Forecast** -sensori tarjoaa yhtenäisen 170 tunnin tuntiennusteen visualisointiin ja hintanäyttöön. Tila on nykyisen tunnin kuluttajahinta EUR/kWh.

**Duration Forecast** -sensori tarjoaa D(k)-matriisin optimointipäätöksiin. Tila on tämän päivän D(4) — keskimääräinen kustannus halvimmille 4 tunnille — nopeana kustannusindikaattorina.

---

## PV-tietoinen hinnoittelu (valinnainen)

Kun käyttäjä asettaa nollasta poikkeavan `pv_capacity_kwp` -arvon (tai määrittää ulkoisen PV-ennuste-entiteetin), koordinaattori lisää jokaiseen ennustetuntiin **marginaalisen efektiivisen hinnan** — paljonko maksaa ajaa yksi lisä-kWh joustavaa kuormaa kyseisellä tunnilla, kun otetaan huomioon kotitalouden PV-tuotto ja tyypillinen kokonaisperustaso.

### Merkinnät

Tunnille `h`:

| Symboli | Merkitys |
|---------|----------|
| `b_h` | Kuluttajan ostohinta (EUR/kWh) = `(spot/1000 + marginaali + siirtohinta + vero) × ALV` |
| `s_h` | Myyntihinta (EUR/kWh) = `spot/1000 − myyntipalkkio − valinn. verkkomaksu`. EI rajoitettu nollaan — voi olla negatiivinen ylituotantoaikoina. |
| `c_h` | Konfiguroitu perustaso (kWh) = `annual_consumption_kwh / 8760 × kuukausikerroin` |
| `p_h` | Tunnittainen PV-tuotto (kWh) sisäisestä estimaattorista tai ulkoisesta entiteetistä. Rajoitettu välille `[0, capacity_kwp · efficiency]`. |

### Marginaalinen efektiivinen hinta (D(k):n syöte)

```
pv_avail_h = max(0, p_h − c_h)
from_pv    = min(1, pv_avail_h)
from_grid  = 1 − from_pv
m_h        = from_pv · s_h + from_grid · b_h
```

Ominaisuudet:

- **Analyyttisesti rajoitettu**: `m_h ∈ [s_h, b_h]` aina.
- **Itsekulutushyöty**: kun PV-ylijäämä ≥ 1 kWh, `m_h = s_h` (luovut vientituotosta, et maksa vähittäishintaa).
- **Osittainen kattavuus**: lineaarinen interpolaatio osto- ja myyntihinnan välillä.
- **Negatiivisen spotin vastuu**: `s_h < 0` siirtyy `m_h:hen` (harvinainen mutta todellinen).
- **Äärellinen kapasiteetti**: `p_h ≤ capacity_kwp · efficiency` on kova katto.

### PV-tietoiset D(k) halpa/kallis -käyrät

Lasketaan suoraan paikallisen päivän 24 tuntittaisesta `effective_eur_kwh` -arvosta käyttäen samaa `compute_dk_cheap_peak`-apufunktiota kuin spot-pohjainen polku. Lajiteltu nousevasti → `dk_cheap_pv[12]` (monotonisesti ei-laskeva); laskevasti → `dk_peak_pv[12]` (monotonisesti ei-nouseva).

**Vahvistettu 4 vuoden todellisella datalla** (1 460 päivää, 5 kWp): nolla monotonisuusrikkomusta, D(1):n keskiarvo 6,90 c/kWh — rajattu, realistinen, optimointiin valmis.

### Vakausinvariantti — avoin silmukka optimoijaa kohden

Ennusteen on pysyttävä deterministisenä funktiona `(spot, sää, PV-konfiguraatio, perustaso-konfiguraatio)` -syötteistä, jotta alavirran optimoijan joustavakuorma-päätökset eivät voi syöttyä takaisin seuraavan syklin hintaennusteeseen. Konkreettisesti:

- **`_resolve_baseload(ts)` EI saa kutsua `hass.states.get` -kutsua tai mitään HA-entiteetinlukua.** Varmistettu grep-testillä tiedostossa `tests/test_coordinator_pv.py`.
- **Konfiguroidun `annual_consumption_kwh`-arvon tulisi edustaa käyttäjän tyypillistä KOKONAISKULUTUSTA vuodessa** — laskupohjaista kokonaiskysyntää mukaan lukien kaikki kuormat (lämpöpumppu, sähköauto, sauna, vedenlämmitin jne.). Staattinen konfiguraatio ei voi luoda optimoijan takaisinkytkentää koska se ei riipu havaitusta kulutuksesta; varsinainen vakausvaatimus koskee vain sitä mitä ennustaja LUKEE HA:sta.
- **`_read_external_pv_forecast()` SAA lukea HA-entiteetin** koska PV-ennuste on säävetoinen ja riippumaton optimoijan päätöksistä — takaisinkytkentää ei muodostu.

#### Miksi "tyypillinen kokonais" eikä "vain ei-joustava" — esimerkkilaskelma

Aurinkoinen keskipäivä, 4 kWh PV, lämpöpumpputalous 16 000 kWh/v tyypillisellä kysynnällä:

| Tilanne | perustaso | pv_avail | m_h | Käyttäytyminen |
|---|---|---|---|---|
| **A: vain ei-joustava (~0,5 kWh/h)** | 0,5 | 3,5 kWh | ≈ 4 c/kWh | Liian optimistinen. Ennuste väittää että kaikki PV on ilmaista lisäkuormalle. EMHASS aikatauluttaa lämpöpumpun + lisäkuormat → toinen kuorma vetääkin 16 c/kWh verkosta. **Systemaattinen optimismivinouma.** |
| **B: tyypillinen kokonais (~1,83 kWh/h × kausi)** | ~1,83 | 2,17 kWh | ≈ 4 c/kWh | Itsestään yhteensopiva. Ennuste olettaa että tyypillinen kysyntä (lämpöpumppu jne.) tapahtuu; EMHASS suunnittelee sen ympärille; todellisuus vastaa oletusta; tasapaino. |

PV:n ollessa vain 2 kWh, Tapaus B palauttaa oikein m_h ≈ 14 c/kWh (PV pääosin tyypillisen kysynnän käyttämä, vain 0,17 kWh marginaalia). Tapaus A olisi palauttanut ~10 c/kWh — silti optimistinen. Molemmat tapaukset täyttävät vakausinvariantin koska molemmat ovat staattista konfiguraatiota; Tapaus B on **tarkempi** koska PV/verkko-suhde joka määrittää marginaalikustannuksen on aidosti kokonaiskysynnän — ei ei-joustavan kysynnän — funktio.

#### Perustasoskeema

PV-tietoinen polku käyttää kahta konfiguraatiokenttää johtaakseen per-tunti-perustason:

- **`annual_consumption_kwh`** (oletus 12 000) — käyttäjän tyypillinen KOKONAISKULUTUS vuodessa sähkölaskun mukaan. Yksi käyttäjäystävällinen luku. Sisäisesti:

  ```
  baseload(h) = annual_consumption_kwh / 8760 × monthly_factor[month_of_h]
  ```

  jossa `monthly_factor` on 12-elementtinen suomalaisen ei-sähkölämmitteisen asuinrakennuksen kausiprofiili `const.py`:ssä (`FINLAND_RESIDENTIAL_MONTHLY_FACTORS`). Kerroinsumma = 12,00 tarkasti (normalisointi-invariantti); vaihteluväli ≈ ±19 % keskiarvon ympärillä (60°N leveysasteen kuvio: valaistuksen vetämä talvihuippu joulu/tammikuussa, loma-ajan/pitkien päivien aallonpohja heinäkuussa). Lähde: kirjallisuuspohjainen estimaatti suomalaisesta tutkimuksesta (VTT Publications 289, Adato Energia, Tilastokeskus 2024). **TODO**: korvaa Fingrid Open Data -aineisto #360:n (BE03 tyyppikäyrä) sanasta-sanaan-arvoilla.

- **`consumption_entity`** (valinnainen) — mikä tahansa HA-kulutusanturi; integraatio tunnistaa tyypin ja tasaa sisäisesti:

  | Tunnistettu tyyppi | Tunnistus (HA-attr) | Tasausstrategia |
  |---|---|---|
  | Kumulatiivinen kWh-laskuri | `unit = kWh`, `state_class = total_increasing` | 14 päivän delta jaettuna 14:llä → päiväkohtainen kWh |
  | Päivä/kuukausi `utility_meter` | `state_class = total` cycle-attribuutilla | Historiaikkunan keskiarvo päiväkohtaisista summista |
  | Hetkellinen teho | `unit = W` tai `kW`, `device_class = power` | `statistics_during_period(28 d, mean)` × 24 |
  | Tuntematon | (varatoiminto) | Hiljainen palautuminen `annual_consumption_kwh` -konfiguraatioon; loki varoitus |

  Tasattu arvo välimuistissa `.storage/spot_price_predictor_consumption_cache.json`:issa, lasketaan uudelleen enintään kerran päivässä. 5 % hysteresis-aluekielto välimuistissa olevalle arvolle estää pieniä anturimelu-värähtelyitä laukaisemasta uudelleen koordinaattorin päivityksiä.

**Vakaustarkistus**:

- **Oletustila** (`consumption_entity = ""`): `baseload(h)` on deterministinen funktio vain `(annual_consumption_kwh, h)` -syötteistä — ei HA-entiteetinlukua, täysin avoin silmukka.
- **HA-anturitila**: 14 päivän liukuva keskiarvo vaimentaa yksittäisen päivän häiriön `1/14 ≈ 7 %`:iin. Yhdistettynä 5 % hysteresis-aluekieltoon, EMHASS:n 5 kWh kuorman uudelleenajoittaminen päivien välillä tuottaa `5/14 ≈ 0,36 kWh` liukuvan muutoksen, vain ~3 % 12 kWh/päivä perustasosta — alueen sisällä, joten välimuistissa oleva perustaso ei muutu ja EMHASS näkee vakaan ennusteen.

### Ulkoinen PV-ennuste — tuetut attribuutti-konventiot

`_read_external_pv_forecast()` on lähde-agnostinen. Automaattisesti tunnistetut attribuutit, prioriteettijärjestyksessä:

| Konventio | Attribuutti | Muoto | Yksikkö | Muunnos |
|---|---|---|---|---|
| 1. Geneerinen ennustelista | `forecast` | list[dict] | kWh | suoraan (avaimet: `pv_kwh`, `kwh`, `energy`, `value`) |
| 2. Forecast.Solar Wh-sanakirja | `wh_hours` | dict {ISO-aikaleima → numero} | Wh | `/ 1000` |
| 3. Forecast.Solar W-sanakirja | `watts` | dict {ISO-aikaleima → numero} | W | `/ 1000` (1h granulariteetti) |
| 4. EMHASS-malli | `irradiance` | list[numero] | W tai kWh | jos suuruus > 50 → W ja `/ 1000`; muutoin kWh |

Kaikki polut palauttavat enintään 168 tunnittaista kWh-arvoa, jotka rajataan välille `[0, capacity_kwp · efficiency]`. Hiljainen palautuminen sisäiseen estimaattoriin, jos entiteetti puuttuu tai mikään konventio ei sovi.

### Konfiguraatio

PV-järjestelmän parametrit asetetaan HA-asennusvelhon valinnaisessa "PV system" -vaiheessa:

| Kenttä | Oletus | Kuvaus |
|--------|--------|--------|
| `pv_capacity_kwp` | 0 (poissa käytöstä) | Asennettu PV-huipputeho |
| `pv_tilt_deg` | 45 | Paneelin kallistus |
| `pv_azimuth_deg` | 180 (etelä) | 0=P, 90=I, 180=E, 270=L |
| `pv_system_efficiency` | 0,85 | DC/AC + likaantuminen + häviöt yhteenlaskettuna |
| `pv_external_entity` | "" | Valinnainen HA-sensori, joka korvaa sisäisen estimaattorin |
| `pv_export_grid_fee` | 0 | Lisämaksu viedystä energiasta (myyntipalkkion yli) |
| `annual_consumption_kwh` | 12 000 | Tyypillinen KOKONAISKULUTUS vuodessa laskun mukaan, mukaan lukien PV-itsekulutus JA optimoijan ohjaamat kuormat. Kerrotaan sisäänrakennetulla suomalaisella asuinrakennuksen kuukausittaisella kausiprofiililla per-tunti-perustasoa varten. |
| `consumption_entity` (valinnainen) | "" | Mikä tahansa HA-kulutusanturi (kumulatiivinen kWh-laskuri, päivä/kuukausi-utility_meter, hetkellinen teho). Integraatio tunnistaa tyypin ja tasaa sisäisesti 14 päivän liukuvalla keskiarvolla + 5 % hysteresiksellä. Suositeltu placeholder: `sensor.energy_yesterday`. |

`pv_capacity_kwp = 0` (oletus) ja tyhjä `pv_external_entity` poistavat kaikki PV-tietoiset tulosteet siististi — integraatio tuottaa tavupareittain identtiset PV:ttömät tulokset.

---

## Home Assistant -integraatio

### Pyhäpäivien ja arkipäivien tunnistus

Malli käyttää arkipäivä/pyhäpäivä-tietoa kysyntämallien valintaan. Kaksi vaihtoehtoa `holidays.ha_workday_integration` -asetuksella.

#### Vaihtoehto A: HA:n Workday-integraatio (suositeltu)

Käyttää Home Assistantin [Workday-integraatiota](https://www.home-assistant.io/integrations/workday/) pyhäpäivien automaattiseen tunnistukseen.

**Asennus:**
1. **Asetukset** → **Laitteet ja palvelut** → **Lisää integraatio** → hae **Workday**
2. Aseta **Maa** arvoon `FI` (Suomi)
3. Integraatio luo `binary_sensor.workday_sensor`

#### Vaihtoehto B: Sisäänrakennettu pyhäpäivälaskuri

Käyttää `finland.yaml` -tiedoston pyhäpäiväsääntöjä. Koulutusputki käyttää aina tätä vaihtoehtoa.

---

## Alueellinen lokalisointi

Järjestelmää ohjaa aluekonfiguraatiotiedosto (`config/regions/finland.yaml`). Uuden alueen tukeminen:

1. **Tunnista säämittauspisteet** — 5-8 sijaintia, painotettuna asennetun kapasiteetin ja väestötiheyden mukaan
2. **Luo uusi YAML-tiedosto** (esim. `sweden.yaml`)
3. **Määrittele paikallinen hinta-API** — ilmainen day-ahead spot-hintarajapinta
4. **Konfiguroi pyhäpäivät** — kiinteät, pääsiäispohjaiset ja erikoissäännöt
5. **Lisää naapurihintalähteet** kestomallia varten
6. **Aseta kuluttajahinnoittelu** — ALV, energiavero, operaattoritariffit
7. **Sovita** uudelleen `spot_price_predictor.retrain_models` -palvelulla alueellisen datan ollessa paikallisesti välimuistissa

Katso englanninkielisestä dokumentaatiosta ([TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md#regional-localization)) tekoälykehotepohja uusien alueiden säämittauspisteiden tunnistamiseen.

---

## Tarkkuus ja uudelleenkoulutus

### Nykyinen päästä päähän -suorituskyky

Alla olevat luvut on otettu uusimmasta v26-vertailusta todellisella FI-datalla ([studies/results/V2_6_1_BENCHMARK.md](studies/results/V2_6_1_BENCHMARK.md)) — kokoonpano, joka on käytössä tällä hetkellä.

**Tuntimalli (spot-pisteen ennuste):**

| Mittari | Arvo |
|---|:---:|
| MAE (h = 24 … 168) | ≈ 10 EUR/MWh |
| R² | ≈ 0,93 |
| Kynnysylityskorkeiden tuntien osumatarkkuus | 98 % |

**Kestomalli (R², per D(k)-indeksi):**

- Sekä `dk_cheap[i]` että `dk_peak[i]` saavuttavat R² ≥ 0,95 kaikilla i = 0 … 23 ([studies/results/V2_5_17_DK_FULL_RANGE.md](studies/results/V2_5_17_DK_FULL_RANGE.md)).
- cheap_4 MAE 4,4 EUR/MWh, peak_4 MAE 6,9 EUR/MWh testijaksolla.

**PV-tietoisen D(k):n vahvistus** (5 kWp viitekokoonpano, 4 vuoden takautuva testaus 1 460 päivällä): nolla PAVA-monotonisuusrikkomusta, PV-tietoisen D(1):n keskiarvo 6,90 c/kWh (keskihajonta 6,0), analyyttisesti rajattu välille `[s_h, b_h]` jokaiselle tunnille. Arvioitu vuosittainen säästö pelkän verkko-D(4):n yli ≈ 600 EUR/v.

### Suositeltu uudelleenkoulutustaajuus

Neljännesvuosittain on järkevä oletus; `RefitMonitor`-kalibroija merkitsee myös jatkuvan ryöminnän 14 vrk ikkunassa, joten käyttäjät voivat sovittaa uudelleen pyynnöstä, kun tuotantoympäristö muuttuu.

### Uudelleenkoulutuksen suorittaminen

`spot_price_predictor.retrain_models` Home Assistant -palvelu sovittaa mukana tulevat artefaktit paikan päällä uudelleen. Developer Tools → Services -valikosta tai automaatiosta:

```yaml
service: spot_price_predictor.retrain_models
data:
  layers: ["seasonal", "spike", "solar"]   # jätä pois sovittaaksesi kaikki kolme
  # fingrid_api_key: "..."                  # tarvitaan vain aurinkokerroksessa
```

Palvelu kirjoittaa kolme JSON-artefaktia atomisesti uudelleen kansioon `custom_components/spot_price_predictor/data/`, ja koordinaattorit lataavat ne automaattisesti uudelleen seuraavalla päivityssyklillä. Valmistuessaan palvelu laukaisee tapahtuman `spot_price_predictor_models_retrained`.

### Avoin kysymys — syötekohtainen tarkkuusvaikutus

Leave-one-out / SHAP-tyyppinen ablaatio nykyiselle Ridge-kerroinjoukolle antaisi mahdollisuuden määrittää kunkin L2-piirteen marginaalivaikutuksen ja arvioida uudelleen syötteet, jotka todettiin aiemmassa toteutettavuustyössä toimiviksi (ydinvoimavaje, SE3:n siirtokapasiteettiproksi). Tämä tutkimus ei kuulu tämän dokumentaatiokierroksen piiriin — se kuuluu erilliseen kokeelliseen haaraan ja yhdistettäisiin `main`-haaraan vain todistusaineiston perusteella.

---

## Projektin rakenne

```
HA-spot-price-predictor/
├── README.md
├── TECHNICAL_GUIDE.md           # Englanninkielinen dokumentaatio
├── TEKNINEN_TOTEUTUS.md         # Tämä dokumentti (suomeksi)
├── INSTALLATION.md              # Vaiheittainen asennusopas
├── config/regions/
│   └── finland.yaml             # Keskitetty konfiguraatio (kaikki parametrit)
├── custom_components/
│   └── spot_price_predictor/    # HA HACS -integraatio
│       ├── __init__.py              # Sisääntulo + palvelujen rekisteröinti (sis. retrain_models)
│       ├── coordinator.py           # Datahaku + V26Pipeline-orkestrointi
│       ├── v26_pipeline.py          # L1+L2+L3+L4-putki + softplus-pohjavyöhyke + DtACI
│       ├── seasonal_decomposition.py # L1-komponenttien sovittaja / haku
│       ├── hourly_calibration.py    # DtACI-biaskorjaaja / viuhka / refit-monitori
│       ├── price_floor.py           # Softplus-pohjavyöhyke
│       ├── solar_clear_sky.py       # Selkeän taivaan × pilvisyys -aurinkomalli
│       ├── retrain.py               # Uudelleenkoulutuksen orkestroija (HA-palvelun backend)
│       ├── sensor.py                # HA-sensorientiteetit
│       ├── api_client.py            # Asynkroniset API-asiakkaat
│       ├── const.py                 # Vakiot ja oletusarvot
│       └── data/
│           ├── seasonal_components_default.json
│           ├── spike_model_default.json
│           ├── solar_submodel_default.json
│           └── finland.yaml
├── ha_dashboard.yaml            # Home Assistant Lovelace -kojelauta (ApexCharts + Mushroom)
├── studies/                     # Sovitusskriptit ja historialliset analyysit
└── tests/                       # Yksikkö- ja integraatiotestit
```
