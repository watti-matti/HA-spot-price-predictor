# Tekninen toteutus: HA-spot-price-predictor (v2.4.0)

Sähkön kuluttajahinnan ja D(k) = CVaR -kestokustannusten ennustaminen Home Assistantiin. Tuottaa 170 tunnin kuluttajahintaennusteen (EUR/kWh) ja 7 vrk D(k) halpa/kallis -kestokäyrät kuormanohjauksen kustannusoptimointiin, käyttäen log-lineaarista Ridge-regressiota fysiikkapohjaisilla piirteillä ja useiden datalähteiden integrointia. Valinnaisesti rikastaa jokaisen ennustetunnin PV-tietoisella marginaalisella efektiivisellä hinnalla `m_h` ja PV-tietoisilla D(k)-käyrillä, kun käyttäjä konfiguroi kotitalouden aurinkopaneelit.

## Arkkitehtuuri

Järjestelmä koostuu kahdesta vaiheesta: **koulutus** (Python, ajetaan ajoittain PC:llä) ja **päättely** (Home Assistant -integraatio, jatkuvasti päällä). v2.3 lisää valinnaisen **päättelyn jälkeisen PV-muunnoksen**, joka ei vaadi uudelleenkoulutusta.

### Koulutusputki (muuttumaton v2.2:sta)

```
Sahkotin API  ──┐
Open-Meteo API ─┼──> Piirteiden käsittely ──> Log-lineaarinen Ridge ──> model_coefs.json
Elpriset API ───┤    (9 merkkivalidoitua       + Tehovenytys            (tunti- ja kesto-
Elering API ────┤     piirrettä v2.2-           + Kestomalli)            mallin kertoimet)
Fingrid API ────┘     karsinnan jälkeen)
```

### Home Assistant -käyttöönotto

```
Open-Meteo  ──┐
Elpriset    ──┼──> Piirrerakentaja ──> Tuntimalli     ──> Spot-/kuluttajahinta-ennuste (170h)
Elering     ──┤    (puhdas Python)     + Kestomalli       + D(k) halpa/kallis (7 vrk)
Fingrid     ──┘                        (puhdas Python)    │
Nord Pool UMM ─────────────────────────┘                  │
                                                          │
                                                          v
                              (v2.3) PV-tietoinen muunnos ┴─> + effective_eur_kwh per tunti
                              [valinnainen]                    + dk_cheap_pv / dk_peak_pv (7 vrk)
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

| Lähde | Haetut alueet | Käytössä v2.2-mallissa? |
|-------|---------------|-------------------------|
| [Elpriset](https://www.elprisetjustnu.se/api/v1/prices) | SE3 | Kyllä — `ar_se3` + `export_potential_se3` |
| [Elpriset](https://www.elprisetjustnu.se/api/v1/prices) | SE1 | Haetaan vain spread-/historiakontekstia varten; `ar_se1` karsittiin v2.2:ssa (kollineaarinen `ar_se3`:n kanssa, koska FI↔SE-siirtoyhteys päättyy SE3-alueelle) |
| [Elering API](https://dashboard.elering.ee/api/nps/price) | EE | Kyllä — `ar_ee` |

AR(2)-mallit sovitetaan naapurihintojen poikkeamiin tuntiprofiileista. Analyysissa vahvistettu vahva autokorrelaatio (viikkotason lag-1 r = 0,54–0,73, suunnan pysyvyys 100 %).

### Valinnainen verkkodata (ilmainen API-avain)

| Lähde | Tarkoitus |
|-------|-----------|
| [Fingrid Open Data](https://data.fingrid.fi) | Ydinvoimatuotanto (#188) `nuclear_x_scarcity`-piirteen interaktiotermiin |

Rekisteröidy ilmaiseksi osoitteessa data.fingrid.fi. Ilman Fingrid-avainta koulutus käyttää vain sää- ja rajat ylittäviä piirteitä, jolloin syntyvä malli jättää `nuclear_x_scarcity` -piirteen pois (8 mukana tulevasta 9:stä). Päättely voi silti ladata mukana tulevan v2.2:n 9-piirteisen mallin — `nuclear_x_scarcity` antaa arvon 0, kun ydinvoimadataa ei syötetä.

### Ydinvoimaseisokkiaikataulu (ilmainen, ei avainta)

| Lähde | Tarkoitus |
|-------|-----------|
| [Nord Pool UMM](https://ummapi.nordpoolgroup.com/messages) | Suunnitellut ydinvoimaseisokit ennustehorisontin kapasiteettiin |

---

## Piirteiden suunnittelu

Koulutusputki voi laskea enintään 17 merkkivalidoitua ehdokaspiirrettä, mutta v2.2:n leave-one-out -redundanssianalyysi osoitti, että **vain 9 niistä tuo itsenäistä signaalia** — loput 8 olivat joko kollineaarisia jäljelle jäävien kanssa tai eivät tuoneet mitattavaa parannusta walk-forward MAE -mittariin. Mukana tuleva v2.2-malli (toimitetaan muuttumattomana myös v2.3:ssa) käyttää tasan näitä 9 piirrettä. Kaikki säädettävät parametrit ovat `config/regions/finland.yaml` -tiedoston `features`-osiossa.

### Mukana tulevat 9 piirrettä

| # | Piirre | Kategoria | Lähde | Etumerkki | Tehtävä |
|---|--------|-----------|-------|:---:|---------|
| 1 | `wind_speed_weighted` | Tarjonta | Open-Meteo (7 kapasiteettipainotettua FI-pistettä) | − | Hinnan päävetäjä — enemmän tuulta, alempi spot |
| 2 | `month_cos` | Kausivaihtelu | Kalenteri | syklinen | Vuotuinen lämmityskuormitus (talven huippu) |
| 3 | `is_holiday` | Kalenteri | Pyhäpäivälaskuri / HA Workday | − | Pyhäpäivät → alempi teollisuuden kysyntä |
| 4 | `hdd_sq` | Lämpökysyntä | Open-Meteo lämpötila | + | `max(0, 17°C − T)²` — kylmän epälineaarinen vahvistus |
| 5 | `wind_log_scarcity` | Tuulen epälineaarisuus | Open-Meteo | + | `log1p(max(0, 8 − tuuli))` — terävä hintapiikki kun tuuli alle 8 m/s |
| 6 | `ar_se3` | Rajat ylittävä | elprisetjustnu.se SE3 | + | AR(2) poikkeama SE3:n tuntityyppiprofiilista, normalisoitu ÷100 — kuvaa FI↔SE3-siirtoyhteyttä |
| 7 | `ar_ee` | Rajat ylittävä | Elering EE | + | AR(2) poikkeama EE:n tuntityyppiprofiilista, normalisoitu ÷100 — kuvaa FI↔EE-yhteyttä |
| 8 | `export_potential_se3` | Rajat ylittävä | SE3-hintaero | − | `max(0, −spread_7d_fi_se3)` — kun FI on halvempi kuin SE3, FI→SE3-vienti nostaa FI-hintaa |
| 9 | `nuclear_x_scarcity` | Ydinvoima | Fingrid #188 + Nord Pool UMM | + | `nuclear_deficit × wind_log_scarcity` — seisokki vahvistaa sääpohjaista niukkuutta |

AR(2)-mallit hajottavat naapurihinnat deterministiseen päiväprofiiliin (arkipäivä vs viikonloppu, 24 tuntia kumpaakin) plus stokastinen AR(2)-poikkeama. AR-poikkeama vaimennetaan (maksimijuuri < 0,95), joten useamman askeleen ennusteet konvergoituvat päiväprofiiliin noin 24 tunnissa, mikä takaa vakauden koko 170 tunnin ennusteikkunassa.

`nuclear_x_scarcity` vaatii ilmaisen Fingrid API-avaimen reaaliaikaiseen ydinvoimadataan; suunnitellut seisokkiaikataulut tulevat [Nord Pool UMM -alustalta](https://umm.nordpoolgroup.com/) (julkinen rajapinta, ei avainta). Ilman Fingrid-avainta koulutus jättää tämän yhden piirteen pois (mallissa 8 piirrettä) ja päättely voi silti ladata mukana tulevan 9-piirteisen mallin — piirre antaa vain arvon 0, kun ydinvoimadataa ei syötetä.

### v2.2:ssa karsitut piirteet (ja syy)

Leave-one-out -pyyhkäisy poisti 8 ehdokasta v2.0/v2.1:n 17-piirteisestä joukosta:

| Karsittu piirre | Syy |
|---|---|
| `solar_irradiance_weighted` | Suomen aurinkoenergian osuus liian pieni vaikuttaakseen spot-hintaan; kerroin oli erottumaton nollasta. |
| `hour_sin`, `hour_cos` | Tunti-vuorokaudessa-kuvio kaappautuu täysin AR(2):n päivätyyppiprofiileihin (`ar_se3` / `ar_ee`). |
| `month_sin` | Vuoden kuukausi kaappautuu riittävästi pelkällä `month_cos`:lla (lämmityshuippu ↔ kosinin ääriarvo). |
| `wind_calm_x_peak_am`, `wind_calm_x_peak_pm` | Voimakkaasti kollineaarisia `wind_log_scarcity`:n kanssa; lisäys-MAE-hyöty oli pyyhkäisyssä negatiivinen. |
| `ar_se1` | Voimakkaasti kollineaarinen `ar_se3`:n kanssa. Fenno-Skan / Fenno-Skan 2 -kaapelit yhdistävät FI:n SE3-alueelle, eivät SE1:lle, joten SE3 hallitsee siirtosignaalia. SE1 haetaan edelleen spread-kontekstia varten mutta ei enää piirteenä. |
| `nuclear_deficit` | Yksinään ydinvoimavajeesta tuli vain vähän hyötyä, kun `nuclear_x_scarcity` (interaktiotermi) säilytettiin. Molempien sisällyttäminen aiheutti multikollineaarisuutta. |

Karsinnan vaikutus (koulutuksen testijako, 4 vuoden historia): MAE 23,94 → 20,07 EUR/MWh (−16 %); R² 0,515 → 0,719 (+40 %). Walk-forward MAE 180 vrk testijaksolla on 20,99 EUR/MWh, selvästi alle pelkän AR(2):n naapurihintaperustason 37,82.

### Mukana tuleva malli vs uudelleenkoulutus: piirremäärä

Mukana tuleva `model_coefs_default.json` on kiinteästi 9-piirteinen v2.2-malli riippumatta siitä mitä rajapintoja päättely tavoittaa. Jos koulutat paikallisesti uudelleen (`python -m src.train_model …`), syntyvä malli käyttää suurinta 9:n osajoukkoa, jonka datalähteesi tukevat:

| Saatavilla oleva data | Koulutetut piirteet | Huomautukset |
|---|:---:|---|
| Vain Open-Meteo | 5 | Pudottaa `ar_se3`, `ar_ee`, `export_potential_se3`, `nuclear_x_scarcity` |
| + elprisetjustnu.se (SE3) + Elering (EE) | 8 | Pudottaa `nuclear_x_scarcity` |
| + Fingrid (ilmainen avain) | **9** | Täysi mukana tulevan mallin piirrejoukko |

---

## Malliarkkitehtuuri

### Tuntimalli: Log-lineaarinen Ridge-regressio

**Ennustekaava:** `hinta = skaala × max(0, exp(Σ kerroin_i × piirre_i + vakio) − log_offset) ^ potenssi`

Log-muunnos käsittelee luonnollisesti epälineaarisen hinta-niukkuussuhteen: lähes lineaarinen matalilla hinnoilla, eksponentiaalinen vahvistus korkeilla hinnoilla.

- Ridge-regressio kohteella `log(hinta + log_offset)` (v2.2 viritti `log_offset`-arvon 55 → 100 sopimaan paremmin 2025–2026 hintaregiimiin)
- **9 merkkivalidoitua piirrettä** v2.2:n mukana tulevassa mallissa (8 ehdokaspiirrettä karsittiin leave-one-out -redundanssipyyhkäisyssä)
- Tehovenytys (`skaala`, `eksponentti`) sovitetaan Nelder-Mead-optimoinnilla testijoukolla
- Aikapainotus: puoliintumisaika 120 päivää (alueittain konfiguroitavissa)
- Ridge α = 50, laajennettu matriisi (ei sakkoa vakiotermille)

**Koulutus:** 4+ vuoden historiallinen data, aikajärjestetty 85/15 jako, eräpohjainen normaaliyhtälöiden ratkaisu (512 riviä). v2.3 toimittaa saman kerroinkartiotiedoston kuin v2.2 — PV-tietoiset tulosteet lasketaan koordinaattorissa, ei koulutetussa mallissa.

### Kestomalli: Segmenttihierarkkinen Ridge + PAVA

Ennustaa D(k) = keskimääräisen spot-hinnan halvimmille k tunnille päivässä. D(k) on matemaattisesti ekvivalentti ehdollisen riskin arvon (CVaR, Conditional Value-at-Risk) kanssa päivänsisäisestä hintajakaumasta tasolla α = k/24, mikä tekee siitä luonnollisen kustannusmittarin kuormien ajoitukseen: "ajoita halvimmille k tunnille" minimoi CVaR:n.

**PAVA** (Pool Adjacent Violators Algorithm) on isotonisen regression menetelmä, joka pakottaa monotonisuuden. Koska D(k) on määritelmän mukaan ei-vähenevä — useampien tuntien lisääminen keskiarvoon voi sisältää vain yhtä kalliita tai kalliimpia tunteja — PAVA yhdistää itsenäisten Ridge-ennusteiden rikkomukset keskiarvoistamalla vierekkäisiä pareja kunnes D(1) ≤ D(2) ≤ ... ≤ D(N) toteutuu kaikkialla.

**Arkkitehtuuri (Phase A halpa/kallis -kaksisuuntainen koulutus):**
- 4 päiväsegmenttiä tariffirajojen mukaan: yö (22-07, 9 tasoa), aamu (07-12, 5 tasoa), keskipäivä (12-18, 6 tasoa), ilta (18-22, 4 tasoa). Yhteensä 24 tuntipaikkaa.
- Jokainen `(segmentti, suunta, k)`: itsenäinen Ridge-malli. Jokainen segmentti sisältää `cheap_models` (k = 1..n_levels) ja `peak_models` (k = 1..n_levels). Mukana tulevia Ridge-sovituksia yhteensä = 2 × (9 + 5 + 6 + 4) = **48 pientä mallia**.
- Segmenttikohtaiset **12 piirrettä** (segmentin tuntien yli laskettuja keskiarvoja):
  `wind_mean`, `solar_mean`, `hdd_mean`, `se3_mean`, `se1_mean`, `nuclear_deficit`, `is_workday`, `month_sin`, `month_cos`, `wind_log_scarcity`, `net_load_mean`, `net_load_squared_mean`. Kaksi viimeistä asetetaan nollaksi kun Fingrid-kuormitusennusteita ei ole saatavilla, vastaten koulutuspuolen oletusarvoa.
- Log-lineaarinen kohde v2.2:n uudelleen viritetyllä offsetilla: `log(D(k) + 100)`
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

**Tuloste:** `model_coefs.json` sisältäen tuntimallin Ridge-kertoimet (9-piirteinen v2.2 mukana tuleva), AR(2)-parametrit `ar_se3`:lle ja `ar_ee`:lle, sekä kaksisuuntaiset halpa/kallis-kestomallin kertoimet (48 segment-suunta-k Ridge-sovitusta).

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
| `forecast` | array[170] | `{timestamp, spot_eur_mwh, consumer_eur_kwh, wind, solar, temp}` per tunti |
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

`daily_forecast`-attribuutti sisältää enintään 7 päivää. Jokaiselle päivälle annetaan sekä uusi (Phase A) että vanha (poistuva) skeema rinnakkain siirtymävaiheen ajan. Anturin tila on tämän päivän `dk_cheap_eur_kwh[3]` — halvimpien 4 tunnin keskihinta.

| Attribuutti | Muoto | Yksikkö | Kuvaus |
|-------------|-------|---------|--------|
| `daily_forecast` | array[≤7] | — | Yksi rivi per päivä |
| `daily_forecast[i].date` | string | — | ISO-päivämäärä (VVVV-KK-PP) |
| `daily_forecast[i].weekday` | string | — | Mon–Sun |
| `daily_forecast[i].source` | string | — | `forecast` tai `actual` (toteutuneet päivät Sahkotinista) |
| **Phase A (suositeltu):** | | | |
| `daily_forecast[i].dk_cheap_eur_kwh` | float[12] | EUR/kWh | Halvimpien k tunnin keskihinta, k=1..12 (ei-laskeva) |
| `daily_forecast[i].dk_peak_eur_kwh`  | float[12] | EUR/kWh | Kalleimpien k tunnin keskihinta, k=1..12 (ei-nouseva) |
| `daily_forecast[i].dk_cheap_spot_eur_mwh` | float[12] | EUR/MWh | Sama spot-hinnoissa |
| `daily_forecast[i].dk_peak_spot_eur_mwh`  | float[12] | EUR/MWh | Sama spot-hinnoissa |
| **Vanha (poistuva):** | | | |
| `daily_forecast[i].dk_consumer_eur_kwh` | float[24] | EUR/kWh | Vanha kumulatiivinen D(k); `dk_consumer_eur_kwh[k-1] == dk_cheap_eur_kwh[k-1]` arvoille k=1..12 |
| `daily_forecast[i].dk_spot_eur_mwh` | float[24] | EUR/MWh | Vanha spot-D(k) |
| `forecast_days` | int | — | Päivien lukumäärä (enintään 7) |

**Käyttötavat:**
- Halvimmat k tuntia päivänä d: `daily_forecast[d].dk_cheap_eur_kwh[k-1]` (siirrettävien kuormien aikataulutus)
- Kalleimmat k tuntia päivänä d: `daily_forecast[d].dk_peak_eur_kwh[k-1]` (varakapasiteetti, pahimman tapauksen suunnittelu)
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
| `c_h` | Konfiguroitu perustaso (kWh) = `baseload_kwh_per_hour × {päivä-/yökerroin}` |
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
- **Konfiguroidun `baseload_kwh_per_hour`-arvon tulisi edustaa käyttäjän tyypillistä KOKONAISTUNTIKULUTUSTA** — laskupohjaista kokonaiskysyntää mukaan lukien kaikki kuormat (lämpöpumppu, sähköauto, sauna, vedenlämmitin jne.). Staattinen konfiguraatio ei voi luoda optimoijan takaisinkytkentää koska se ei riipu havaitusta kulutuksesta; varsinainen vakausvaatimus koskee vain sitä mitä ennustaja LUKEE HA:sta.
- **`_read_external_pv_forecast()` SAA lukea HA-entiteetin** koska PV-ennuste on säävetoinen ja riippumaton optimoijan päätöksistä — takaisinkytkentää ei muodostu.

#### Miksi "tyypillinen kokonais" eikä "vain ei-joustava" — esimerkkilaskelma

Aurinkoinen keskipäivä, 4 kWh PV, lämpöpumpputalous 16 000 kWh/v tyypillisellä kysynnällä:

| Tilanne | perustaso | pv_avail | m_h | Käyttäytyminen |
|---|---|---|---|---|
| **A: vain ei-joustava (~0,5 kWh/h)** | 0,5 | 3,5 kWh | ≈ 4 c/kWh | Liian optimistinen. Ennuste väittää että kaikki PV on ilmaista lisäkuormalle. EMHASS aikatauluttaa lämpöpumpun + lisäkuormat → toinen kuorma vetääkin 16 c/kWh verkosta. **Systemaattinen optimismivinouma.** |
| **B: tyypillinen kokonais (~1,83 kWh/h × kausi)** | ~1,83 | 2,17 kWh | ≈ 4 c/kWh | Itsestään yhteensopiva. Ennuste olettaa että tyypillinen kysyntä (lämpöpumppu jne.) tapahtuu; EMHASS suunnittelee sen ympärille; todellisuus vastaa oletusta; tasapaino. |

PV:n ollessa vain 2 kWh, Tapaus B palauttaa oikein m_h ≈ 14 c/kWh (PV pääosin tyypillisen kysynnän käyttämä, vain 0,17 kWh marginaalia). Tapaus A olisi palauttanut ~10 c/kWh — silti optimistinen. Molemmat tapaukset täyttävät vakausinvariantin koska molemmat ovat staattista konfiguraatiota; Tapaus B on **tarkempi** koska PV/verkko-suhde joka määrittää marginaalikustannuksen on aidosti kokonaiskysynnän — ei ei-joustavan kysynnän — funktio.

#### v2.3 → v2.3.1 dokumentaatiokorjaus

v2.3.0 -julkaisu sisälsi virheellisen ohjeen "vain ei-joustava". **Ohje oli väärä** — se sekoitti kaksi erillistä vakaushuolta. Varsinainen vaatimus koskee vain sitä mitä ennustaja LUKEE (ei optimoijan vaikuttamia HA-entiteettejä), ei sitä mitä staattinen arvo EDUSTAA. Vanhaa ohjetta noudattavat käyttäjät saavat Tapauksen A optimismivinouman lämpöpumppupäivinä. Nosta `baseload_kwh_per_hour` tyypilliseen KOKONAISKULUTUKSEEN (≈ vuosilasku_kWh / 8760).

#### v2.4.0 perustasoskeemauudistus

v2.4.0 korvaa kolme v2.3-kenttää (`baseload_kwh_per_hour`, `baseload_day_factor`, `baseload_night_factor`) kahdella ystävällisemmällä:

- **`annual_consumption_kwh`** (oletus 12 000) — käyttäjän tyypillinen KOKONAISKULUTUS vuodessa sähkölaskun mukaan. Yksi käyttäjäystävällinen luku. Sisäisesti:

  ```
  baseload(h) = annual_consumption_kwh / 8760 × monthly_factor[month_of_h]
  ```

  jossa `monthly_factor` on 12-elementtinen suomalaisen ei-sähkölämmitteisen asuinrakennuksen kausiprofiili `const.py`:ssä (`FINLAND_RESIDENTIAL_MONTHLY_FACTORS`). Kerroinsumma = 12,00 tarkasti (normalisointi-invariantti); vaihteluväli ≈ ±19 % keskiarvon ympärillä (60°N leveysasteen kuvio: valaistuksen vetämä talvihuippu joulu/tammikuussa, loma-ajan/pitkien päivien aallonpohja heinäkuussa). Lähde: kirjallisuuspohjainen estimaatti suomalaisesta tutkimuksesta (VTT Publications 289, Adato Energia, Tilastokeskus 2024). **TODO**: korvaa Fingrid Open Data -aineisto #360:n (BE03 tyyppikäyrä) sanasta-sanaan-arvoilla v2.4.x-päivityksessä.

- **`consumption_entity`** (valinnainen) — mikä tahansa HA-kulutusanturi; integraatio tunnistaa tyypin ja tasaa sisäisesti:

  | Tunnistettu tyyppi | Tunnistus (HA-attr) | Tasausstrategia |
  |---|---|---|
  | Kumulatiivinen kWh-laskuri | `unit = kWh`, `state_class = total_increasing` | 14 päivän delta jaettuna 14:llä → päiväkohtainen kWh |
  | Päivä/kuukausi `utility_meter` | `state_class = total` cycle-attribuutilla | Historiaikkunan keskiarvo päiväkohtaisista summista |
  | Hetkellinen teho | `unit = W` tai `kW`, `device_class = power` | `statistics_during_period(28 d, mean)` × 24 |
  | Tuntematon | (varatoiminto) | Hiljainen palautuminen `annual_consumption_kwh` -konfiguraatioon; loki varoitus |

  Tasattu arvo välimuistissa `.storage/spot_price_predictor_consumption_cache.json`:issa, lasketaan uudelleen enintään kerran päivässä. 5 % hysteresis-aluekielto välimuistissa olevalle arvolle estää pieniä anturimelu-värähtelyitä laukaisemasta uudelleen koordinaattorin päivityksiä.

**Vakaustarkistus v2.4-skeemalla**:

- **Oletustila** (`consumption_entity = ""`): `baseload(h)` on deterministinen funktio vain `(annual_consumption_kwh, h)` -syötteistä — ei HA-entiteetinlukua, täysin avoin silmukka. Identtinen turvaominaisuus Phase 1:n kanssa.
- **HA-anturitila**: 14 päivän liukuva keskiarvo vaimentaa yksittäisen päivän häiriön `1/14 ≈ 7 %`:iin. Yhdistettynä 5 % hysteresis-aluekieltoon, EMHASS:n 5 kWh kuorman uudelleenajoittaminen päivien välillä tuottaa `5/14 ≈ 0,36 kWh` liukuvan muutoksen, vain ~3 % 12 kWh/päivä perustasosta — alueen sisällä, joten välimuistissa oleva perustaso ei muutu ja EMHASS näkee vakaan ennusteen.

**Migraatio v2.3.x:stä**: kun konfiguraatiomerkintä sisältää vain vanhan `baseload_kwh_per_hour` -kentän, koordinaattorin `__init__` päättelee vastaavan vuosittaisen arvon ja kirjaa INFO-rivin. Vanhat kentät säilyvät `entry.data`:ssa kunnes käyttäjä avaa Asetukset-dialogin ja tallentaa uudelleen. v2.3.0:n oletusta noudattanut käyttäjä saa noin 7660 kWh/v päättelyn — selvästi alhainen tyypilliselle suomalaiselle lämpöpumpputalolle, mikä kannustaa virittämään todelliseen laskuun.

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
| `annual_consumption_kwh` (v2.4) | 12 000 | Tyypillinen KOKONAISKULUTUS vuodessa laskun mukaan, mukaan lukien PV-itsekulutus JA optimoijan ohjaamat kuormat. Kerrotaan sisäänrakennetulla suomalaisella asuinrakennuksen kuukausittaisella kausiprofiilla per-tunti-perustasoa varten. |
| `consumption_entity` (v2.4, valinnainen) | "" | Mikä tahansa HA-kulutusanturi (kumulatiivinen kWh-laskuri, päivä/kuukausi-utility_meter, hetkellinen teho). Integraatio tunnistaa tyypin ja tasaa sisäisesti 14 päivän liukuvalla keskiarvolla + 5 % hysteresiksellä. Suositeltu placeholder: `sensor.energy_yesterday`. |
| `baseload_kwh_per_hour` (v2.3, vanha) | 0,8 | Auto-migroidaan `annual_consumption_kwh`-arvoon latauksessa. Säilytetty taaksepäin yhteensopivuuden vuoksi. |
| `baseload_day_factor` (v2.3, vanha) | 1,2 | Auto-migroidaan; sivuutetaan v2.4:ssä. |
| `baseload_night_factor` (v2.3, vanha) | 0,7 | Auto-migroidaan; sivuutetaan v2.4:ssä. |

`pv_capacity_kwp = 0` (oletus) ja tyhjä `pv_external_entity` poistavat kaikki PV-tietoiset tulosteet siististi — integraatio tuottaa tavupareittain identtiset PV:ttömät tulokset, jotka vastaavat v2.2:n käyttäytymistä.

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
5. **Lisää naapurihintalähteet** rajat ylittäville piirteille
6. **Aseta kuluttajahinnoittelu** — ALV, energiavero, operaattoritariffit
7. **Aja koulutus** komennolla `--region sweden`

Katso englanninkielisestä dokumentaatiosta ([TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md#regional-localization)) tekoälykehotepohja uusien alueiden säämittauspisteiden tunnistamiseen.

---

## Tarkkuus ja uudelleenkoulutus

### Nykyinen suorituskyky (v2.3.0 — mukana tuleva v2.2:n 9-piirteinen karsittu malli, 4 vuoden koulutusdata)

**Tuntimalli:**

| Mittari | v2.1 (17 piirrettä) | v2.2 / v2.3 (9 piirrettä) | Muutos |
|---|:---:|:---:|:---:|
| MAE (koulutuksen testijako) | 23,94 EUR/MWh | **20,07 EUR/MWh** | −16 % |
| R² | 0,515 | **0,719** | +40 % |
| Walk-forward MAE (180 vrk testijakso) | — | **20,99 EUR/MWh** | vs. AR(2)-perustaso 37,82 |

**Kestomalli (Spearmanin ρ, viimeiset 365 päivää):**

| D(k) | Käyttötapaus | ρ |
|:---:|:-:|:---:|
| D(4) | Halvimmat 4h | 0,930 |
| D(8) | Halvimmat 8h | 0,937 |
| D(24) | Päivän keskiarvo | 0,940 |

**v2.3 PV-tietoisen D(k):n vahvistus** (päättelyn jälkeinen muunnos, ei uudelleenkoulutusta; 5 kWp / 1 kWh-h perustasoa, 4 vuoden takautuva testaus 1 460 päivällä): nolla PAVA-monotonisuusrikkomusta, PV-tietoisen D(1):n keskiarvo 6,90 c/kWh (keskihajonta 6,0), analyyttisesti rajattu välille `[s_h, b_h]` jokaiselle tunnille. Arvioitu vuosittainen säästö pelkän verkko-D(4):n yli ≈ 600 EUR/v.

### Suositeltu uudelleenkoulutustaajuus

**Kouluta uudelleen 3-4 kuukauden välein (neljännesvuosittain).**

### Uudelleenkoulutuksen suorittaminen

```bash
cd HA-spot-price-predictor
pip install -r requirements.txt

# Kouluta uusimmalla datalla
export FINGRID_API_KEY=avaimesi  # valinnainen
python -m src.train_model --region finland --fingrid-key YOUR_KEY

# Luo seurantakojelauta
python model_dashboard.py

# Luo 7 vrk ennustekojelauta
python forecast_dashboard.py

# Kopioi kertoimet HA-integraatioon
cp output/model_coefs.json custom_components/spot_price_predictor/data/model_coefs_default.json
```

### Puoliintumisaika-parametri

`half_life_days` (oletus: 120) määrittää miten malli painottaa historiallista dataa koulutettaessa. 120 päivää vanha data painolla 50%. Optimoitu Suomen nopeasti kasvavalle tuulivoimakapasiteetille.

Kestomalli käyttää erillistä unohtamiskerrointa λ = 0,960 (puoliintumisaika 17 päivää), optimoitu sääregiimimuutosten seuraamiseen.

---

## Projektin rakenne

```
HA-spot-price-predictor/
├── README.md
├── TECHNICAL_GUIDE.md           # Englanninkielinen dokumentaatio
├── TEKNINEN_TOTEUTUS.md         # Tämä dokumentti (suomeksi)
├── config/regions/
│   └── finland.yaml             # Keskitetty konfiguraatio (kaikki parametrit)
├── src/
│   ├── train_model.py           # Koulutusputki
│   ├── features.py              # Piirteiden käsittely (koulutus)
│   ├── data_sources.py          # API-asiakasohjelmat (koulutus)
│   └── holidays.py              # Pyhäpäivälaskuri
├── custom_components/
│   └── spot_price_predictor/    # HA HACS -integraatio
│       ├── model.py             # Puhdas Python päättely (tunti + kesto)
│       ├── features.py          # Puhdas Python piirrerakentaja
│       ├── coordinator.py       # HA-datakoordinaattori
│       ├── sensor.py            # HA-sensorientiteetit
│       ├── api_client.py        # Asynkroniset API-asiakkaat
│       ├── const.py             # Vakiot ja oletusarvot
│       └── data/
│           ├── model_coefs_default.json  # Esiasennettu malli
│           └── finland.yaml              # Esiasennettu konfiguraatio
├── ha_dashboard.yaml            # Home Assistant Lovelace -kojelauta (ApexCharts + Mushroom)
├── model_dashboard.py           # Seurantakojelauta
├── forecast_dashboard.py        # Ennustekojelauta
├── studies/                     # Arkistoidut analyysiskriptit
├── tests/                       # 267 yksikkötestiä (33 PV-tietoista v2.3:ssa)
└── output/                      # Tuotetut artefaktit
    ├── model_coefs.json
    ├── model_dashboard.html
    └── forecast.html
```
