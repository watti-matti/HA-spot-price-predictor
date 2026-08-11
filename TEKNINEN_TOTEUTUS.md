# Tekninen toteutus — HA Spot Price Predictor

Suomalaisen kuluttajan sähkön spot- ja kuluttajahinnan sekä D(k)-kestokäyrien ennustaminen Home Assistantiin. Tuottaa 170 tunnin pisteennusteen, P5/P25/P50/P75/P95-viuhkavyöt ja seitsemän vuorokauden halpa/kallis-kestokäyrät nelitasoisesta ennusteputkesta.

Tämä on [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md):n suomennos. Se kattaa saman sisällön; pelkät attribuuttiluettelot (kenttien nimet ovat joka tapauksessa englanniksi) on jätetty alkuperäiseen, jotta dokumentit eivät eriydy toisistaan. **Ristiriitatilanteessa englanninkielinen versio ratkaisee.**

## Arkkitehtuuri

Integraatiossa on kaksi vaihetta:

- **Päättely** ajetaan Home Assistantin sisällä. `SpotPriceCoordinator` ([`coordinator.py`](custom_components/spot_price_predictor/coordinator.py)) ohjaa jaksottaisen päivityskierroksen: hakee sää- ja hintadatan, kutsuu `Pipeline`-luokkaa ([`pipeline.py`](custom_components/spot_price_predictor/pipeline.py)) nelitasoiseen ennusteeseen, rakentaa tunti- ja päiväkohtaiset attribuutit ja työntää ne sensorientiteeteille ([`sensor.py`](custom_components/spot_price_predictor/sensor.py)).
- **Uudelleenkoulutus** on pyynnöstä tehtävä `data/`-kansion kolmen JSON-artefaktin uudelleensovitus, joka on julkaistu palveluna `spot_price_predictor.retrain_models`. Sovitusskriptit (`studies/build_seasonal_components.py`, `studies/build_fresh_spike_model.py`, `studies/solar_clear_sky_submodel.py`) lukevat välimuistissa olevia parquet-tiedostoja ja kirjoittavat artefaktit atomisesti; koordinaattori lataa ne uudelleen seuraavalla kierroksella.

Ennusteputki lukee kolme jäädytettyä artefaktia kansiosta `custom_components/spot_price_predictor/data/`:

| Artefakti | Ladataan | Sisältö |
|---|---|---|
| `seasonal_components_default.json` | `Pipeline._seasonal_artifact` | Sarjakohtaiset additiiviset komponentit (tunti + päivä + viikko) FI-spotille ja lämpötilalle. |
| `spike_model_default.json` | `Pipeline._spike_artifact` | L2:n `ridge_coef`, L3:n `ar1_phi`, L4:n `gpd_left` / `gpd_right` -häntäparametrit sekä normaalirungon `stats.eta_train_*`. |
| `solar_submodel_default.json` | (vain PV-tietoinen polku) | Selkeän taivaan × pilvisyys -aurinkotuotantomalli. |

Kalibraattoreiden pysyvä tila sijaitsee polussa `<config>/.storage/spot_price_predictor_pipeline/`.

## Nelitasoinen ennusteputki

Julkinen sisääntulo: `Pipeline.compute_forecast(timestamps, wind, solar, temp, recent_fi_residuals=None, neighbour_prices_lag168=None, netload_lag168=None, is_holiday=None, enable_fan_chart=True)`. Jokaisella ennustetunnilla h:

```
mean(h)  = L1 seasonal_fi(h)
         + L2 ridge(h)
         + L3 φ^h · η(t₀−1)
mean(h)  = softplus_floor(mean(h), lattia = −5 EUR/MWh)
mean(h) -= per_hour_bias_corrector.bias_estimate[tunti]

P{5,25,50,75,95}_eur_mwh(h) ← 500 näytteen sekoitus (normaalirunko + GPD-hännät)
```

### L1 — Kausivaihtelun hajotelma

`Pipeline._seasonal_fi` ja `_deseasonalize_input` lukevat additiiviset tunti + päivä + viikko -komponentit ja vähentävät ne FI-hinnasta ja lämpötilasta, jolloin syntyvät kausitasoitetut residuaalit. Tuulelle ja auringolle käytetään spike-artefaktin `physics_seasonal`-lohkon omia komponentteja, jotta koulutus ja päättely tekevät saman muunnoksen. Vanhemmat artefaktit, joista lohko puuttuu, palautuvat paikalliseen keskiarvokeskitykseen.

### L2 — Kausivaihtelusta puhdistetun osan Ridge-regressio (yhdeksän piirrettä)

Toimitettava `spike_model_default.json` sisältää kanonisen piirrejärjestyksen `ridge_features`-kentässään; putki lukee sen rakennusvaiheessa. Varakenttänä on `RIDGE_FEATURES`-vakio.

| # | Piirre | Määritelmä |
|---|---|---|
| 1 | `intercept` | vakio 1 |
| 2 | `Y_fi_lag168` | Kausitasoitettu FI-residuaali 7 vrk aiemmin. Koordinaattori välittää tällä hetkellä nollia (kylmäkäynnistys). |
| 3 | `is_workday` | `weekday < 5` **paikallisessa** (Europe/Helsinki) kalenterissa, pyhäpäivät poistettuna |
| 4 | `Y_sigmoid_wind_rho` | `σ((tuuli − 7,5) / 1,5) × ρ(T) / 1,225` |
| 5 | `Y_solar_effective` | `GHI × (1 − 0,004 · max(0, T_cell − 25))`, `T_cell = T + 0,03 · GHI` |
| 6 | `Y_temp` | Kausitasoitettu lämpötila |
| 7 | `Y_se1_lag168` | Kausitasoitettu SE1-spot **168 h aiemmin** |
| 8 | `Y_se3_lag168` | Sama 168 h viiveellä (FennoSkan-kaapelin Ruotsin pää) |
| 9 | `Y_ee_lag168` | Sama 168 h viiveellä (Estlink-kaapelin Viron pää) |
| 10 | `is_holiday` | Pyhäpäivälippu paikallisen päivämäärän mukaan |

Ridge-ennuste on `ridge = X @ self._ridge_coef`. Kentässä `ridge_coef` on **10** arvoa ja **vakiotermi on ensimmäisenä**, kun taas `ridge_features` ei sisällä sitä — jos listat kohdistaa väärin, tuulen kerroin näyttää auringon kertoimelta.

Naapurihinnat välitetään parametrilla `neighbour_prices_lag168`, jossa alkio *i* on hinta hetkellä `t − 168 h`. **Saman tunnin hintoja ei saa koskaan välittää**: FI, SE1, SE3 ja EE selviävät samassa day-ahead-huutokaupassa, joten saman tunnin naapurihintaa ei voi havaita ennen ennustettavaa kohdetta. Puuttuvat tai NaN-arvot palautuvat nollasarakkeeksi kyseiselle tunnille.

Ajonaikaiset invariantit:

- **168 tunnin naapuriviive** — testi `test_artifact_declares_no_same_hour_neighbour_features` hylkää artefaktin, joka nimeää viivästämättömän vyöhykkeen.
- **Tuulen ja auringon kertoimet ≤ 0** — nollarajakustannuksinen tuotanto ei voi nostaa hintaa. Kouluttaja sovittaa rajoitteen alaisena, ja `Pipeline._enforce_physics_signs` leikkaa positiivisen kertoimen latausvaiheessa.

### L3 — AR(1)-momentum

`Pipeline._ar_contribution` palauttaa `φ^h · last_eta`, kun h = 1..n. `φ` luetaan artefaktin `ar1_phi`-kentästä ja `last_eta` on viimeisin havaittu AR-jäännös, jota `Pipeline.update_with_actuals` päivittää. Kylmäkäynnistyksessä L3:n osuus on nolla.

### Softplus-lattia ja tuntikohtainen harhakorjaus

Ennen viuhkan näytteistystä keskiarvo rajataan −5 EUR/MWh:iin softplus-funktiolla. Tämän jälkeen `PerHourBiasCorrector` vähentää harha-arvion: 24 erillistä EMA-suodinta, yksi kutakin UTC-tuntia kohti, kukin noin yksi havainto vuorokaudessa.

Korjain käyttää **3 vuorokauden puoliintumisaikaa**, kahden havainnon suojarajaa ja CMA→EMA-lämmittelyä (`adaptive_init`). Aiempi 14 vuorokauden puoliintumisaika 14 päivityksen portin takana kytki korjauksen pois täsmälleen yhden puoliintumisajan ajaksi ja otti sen sitten käyttöön 50 %:n voimakkuudella, koska nollasta alkava EMA saavuttaa vain osuuden `1−(1−λ)ⁿ` todellisesta harhasta. Vaimenevalla vahvistuksella `α_n = max(1/n, λ)` arvio on harhaton jokaisella n:n arvolla. Tuottaja: `studies/bias_corrector_warmup_study.py`.

### L4 — GPD POT -piikkimalli

`Pipeline._sample_fan_chart` arpoo 500 polkua sekoituksesta, jossa on normaalirunko ja yleistetyn Pareto-jakauman oikea/vasen häntä. Tuntikohtaiset persentiilit muodostavat kentät `P5_eur_mwh` … `P95_eur_mwh`.

### Pysyvät kalibraattorit

| Tilatiedosto | Luokka | Oletusarvot |
|---|---|---|
| `hourly_bias.json` | `PerHourBiasCorrector` | halflife_days = 3, warmup_updates = 2, adaptive_init = true |
| `hourly_fan_chart.json` | `HourlyFanChartCalibrator` | target_coverages = (0,5, 0,9), window = 720, min_warmup = 24 |
| `refit_monitor.json` | `RefitMonitor` | target_coverage = 0,9, drift_pp = 0,05, persistence_steps = 14 × 24 |

`refit_recommended` nousee, kun toteutunut peitto poikkeaa tavoitteesta yli 5 prosenttiyksikköä 14 vuorokauden ajan.

## Syötteet, joita ei tällä hetkellä käytetä

Koordinaattori hakee useita virtoja, jotka eivät syötä käyttäjälle näkyvää spot-ennustetta: Sähkötinin naapuri- ja historiahinnat, Fingrid #188 (ydinvoimatuotanto), Fingrid #165 / #246 / #247 (kulutus-, tuuli- ja aurinkoennusteet) sekä Nord Poolin UMM-huoltokatkoaikataulu. Ne syötetään vanhaan kestomalliin ja diagnostiikkaan, eivät `RIDGE_FEATURES`-piirteisiin.

## Kestomalli — neljä D(k)-taulukkoa vuorokaudessa

Jokaiselle paikalliselle vuorokaudelle, jolta ennusteikkunassa on 24 täyttä tuntia:

1. **Spot-puoli** — 24 tuntiarvoa lajitellaan nousevaan ja laskevaan järjestykseen, minkä jälkeen otetaan kumulatiiviset keskiarvot: `dk_cheap_eur_mwh[24]` (i+1 halvimman tunnin keskiarvo) ja `dk_peak_eur_mwh[24]` (i+1 kalleimman tunnin keskiarvo).
2. **Kuluttajapuoli** — tuntihinnat muunnetaan ensin kuluttajahinnaksi tuntikohtaisella siirtotariffilla (`day_rate` klo 07–22, `night_rate` klo 22–07), minkä jälkeen sama lajittelu ja kumulatiivinen keskiarvo tuottavat kentät `dk_cheap_eur_kwh[24]` / `dk_peak_eur_kwh[24]`.

Kaikki neljä taulukkoa ovat 0-indeksoituja: `dk_cheap_eur_mwh[3]` on vuorokauden neljän halvimman tunnin keskiarvo. Koko vuorokauden keskiarvo löytyy indeksistä 23 kummastakin suunnasta.

Menneille päiville sama laskenta tehdään toteutuneista Sähkötin-hinnoista ja merkintä saa kentän `source: "actual"`.

## PV-tietoinen tehollinen hinta (valinnainen)

Kun `pv_capacity_kwp > 0` tai ulkoinen PV-ennuste-entiteetti on määritetty, jokainen ennustetunti saa kentän `effective_eur_kwh` — yhden lisä-kWh:n rajakustannus, kun oma PV-tuotanto otetaan huomioon. **Itse käytetty PV lasketaan ilmaiseksi** (ei spot-, siirto- eikä verokomponenttia): `effective_eur_kwh` on vähintään `0` silloin, kun ylijäämä-PV riittää kattamaan lisäkuorman, ja menee negatiiviseksi vain negatiivisilla myyntihinnoilla.

## DtACI-kalibrointi (valinnainen)

Kun `enable_dtaci_dk = true`, koordinaattori ympäröi D(k)-käyrät adaptiivisilla konformaalivöillä: 48 DtACI-instanssia (vain FI-vyöhyke), yksi kutakin paria `(suunta, k)` kohti. Vyöt kirjoitetaan 24-alkioisina taulukkoina päiväkohtaiseen merkintään vasta, kun instanssit ovat lämmenneet; sitä ennen vyöt supistuvat pisteennusteeksi — tarkoituksella, jotta perusteetonta luottamusta ei synny. Lämpenemiseen kuluu noin viisi vuorokautta.

Algoritmin yksityiskohdat: [docs/dtaci_layer.md](docs/dtaci_layer.md).

## PV-tietoinen riski

Kun PV on käytössä, jokainen `daily_forecast[i]` sisältää neljä kenttää: `pv_aware_cvar95_eur_kwh` (tehollisen kustannuksen häntäkeskiarvo huonoimmassa 5 %:ssa yhteisistä hinta- ja PV-skenaarioista), `pv_aware_self_consumed_kwh`, `pv_aware_exported_kwh` ja `pv_aware_data_provenance`. Laskenta arpoo 200 lognormaalisti häirittyä PV-polkua ja ajaa jaetun kustannusytimen jokaiselle polulle.

PV-netotetut `dk_*_pv_eur_kwh`-taulukot ovat **diagnostisia likiarvoja** kojelautoja varten. Kuormakohtaisten optimoijien tulee muodostaa oma α:nsa tuntikohtaisista `consumer_eur_kwh`- ja `sell_eur_kwh`-arvoista.

### Kulutusprofiilin lataaja

Kaksi lähdettä: ulkoinen EMA-moduuli (jos `consumption_profile_entity` on asetettu) tai synteettinen suomalainen kotitalousprofiili, joka on kalibroitu `annual_consumption_kwh`-arvoon. Synteettinen varajärjestely **ei** perustu kenenkään yksittäisen käyttäjän dataan; tietosuojasopimus on kuvattu tiedostossa [docs/household_profile_schema.md](docs/household_profile_schema.md).

## Mittaustarkkuus

Toistamalla käytössä oleva ennusteputki data-varastoa vasten aikavälillä 2023-01 … 2026-07 (30 989 tuntia, toteutunut keskihinta 50,5 EUR/MWh):

| tila | MAE | harha | R² |
|---|--:|--:|--:|
| tuore asennus, ei kalibrointihistoriaa | 25,8 EUR/MWh | +2,1 | 0,47 |
| kalibraattorit lämmenneet | **24,1 EUR/MWh** | −0,5 | 0,51 |

Tiukasti vuotottomassa arvioinnissa — vain tunnit, joita day-ahead-huutokauppa ei ole vielä julkaissut — luku on **27,1 EUR/MWh**. Käytä sitä rehellisenä otoksen ulkopuolisena lukuna.

> Aiemmat julkaisut ilmoittivat lämmenneeksi tarkkuudeksi ≈ 10 EUR/MWh ja R² ≈ 0,91. Luku oli peräisin otoksen sisäisestä sovituksesta ja takatestistä, jonka rajasiirtopiirteet vuosivat kohdemuuttujan. Kalibraattoreiden lämpeneminen on arvoltaan noin 1,7 EUR/MWh ja poistaa jäljelle jäävän harhan — ei 12 EUR/MWh.

MAE ~24 keskihinnalla ~50 tarkoittaa, että ennuste on **järjestystyökalu**, ei hintaoraakkeli. Vuorokauden sisäinen hintahaitari on rutiininomaisesti yli 50 EUR/MWh, joten halpojen ja kalliiden tuntien järjestys on luotettava selvästi ennen absoluuttista tasoa.

## Sensoriattribuutit, palvelut ja päivitystahti

Attribuuttien täydellinen kenttäluettelo, Home Assistant -palvelut, päivitystahti ja projektirakenne: [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md). Kenttien nimet ovat englanninkielisiä tunnisteita, joten niitä ei käännetä.
