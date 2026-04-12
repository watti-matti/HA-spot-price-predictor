# Tekninen toteutus: HA-spot-price-predictor

Suomen pörssisähkön spot-hinnan ennustaminen tunnin tarkkuudella Home Assistantiin log-lineaarisella Ridge-regressiolla käyttäen fysiikkapohjaisia piirteitä, kestokäyräennustetta ja useiden datalähteiden integrointia.

## Arkkitehtuuri

Järjestelmä koostuu kahdesta vaiheesta: **koulutus** (Python, ajetaan ajoittain PC:llä) ja **päättely** (Home Assistant -integraatio, jatkuvasti päällä).

### Koulutusputki

```
Sahkotin API  ──┐
Open-Meteo API ─┼──> Piirteiden käsittely ──> Log-lineaarinen Ridge ──> model_coefs.json
Elpriset API ───┤    (17 validoitua            + Tehovenytys            (tunti- ja kesto-
Elering API ────┤     piirrettä)                + Kestomalli)            mallin kertoimet)
Fingrid API ────┘ (valinnainen)
```

### Home Assistant -käyttöönotto

```
Open-Meteo  ──┐
Elpriset    ──┼──> Piirrerakentaja ──> Tuntimalli     ──> Kuluttajahinta
Elering     ──┤    (puhdas Python)     + Kestomalli       + Halvimmat tunnit
Fingrid     ──┘                        (puhdas Python)    + 7 vrk ennuste
Nord Pool UMM ─────────────────────────┘                  + Kojelauta
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

| Lähde | Alueet | Tarkoitus |
|-------|--------|-----------|
| [Elpriset](https://www.elprisetjustnu.se/api/v1/prices) | SE1, SE3 | Ruotsin spot-hinnat AR-malleihin |
| [Elering API](https://dashboard.elering.ee/api/nps/price) | EE | Viron spot-hinnat AR-malleihin |

AR(2)-mallit sovitetaan naapurihintojen poikkeamiin tuntiprofiileista. Analyysissa vahvistettu vahva autokorrelaatio (viikkotason lag-1 r=0,54-0,73, suunnan pysyvyys 100%).

### Valinnainen verkkodata (ilmainen API-avain)

| Lähde | Tarkoitus |
|-------|-----------|
| [Fingrid Open Data](https://data.fingrid.fi) | Ydinvoimatuotanto (#188) nuclear_deficit- ja niukkuuspiirteisiin |

Rekisteröidy ilmaiseksi osoitteessa data.fingrid.fi. Ilman avainta malli koulutetaan Taso 1+2 -piirteillä (15 piirrettä).

### Ydinvoimaseisokkiaikataulu (ilmainen, ei avainta)

| Lähde | Tarkoitus |
|-------|-----------|
| [Nord Pool UMM](https://ummapi.nordpoolgroup.com/messages) | Suunnitellut ydinvoimaseisokit ennustehorisontin kapasiteettiin |

---

## Piirteiden suunnittelu

Malli v2.0 käyttää 17 merkkivalidoitua piirrettä. Kaikki säädettävät parametrit ovat `config/regions/finland.yaml` -tiedoston `features`-osiossa.

### Taso 1: Peruspiirteet (11) — ei API-avaimia tarvita

| Kategoria | Piirteet | Kertoimen etumerkki |
|-----------|----------|:---:|
| Tarjonta | `wind_speed_weighted`, `solar_irradiance_weighted` | negatiivinen (enemmän tarjontaa = alempi hinta) |
| Aikajaksot | `hour_sin`, `hour_cos`, `month_sin`, `month_cos` | syklinen |
| Kalenteri | `is_holiday` | negatiivinen (vähäisempi kysyntä) |
| Lämpökysyntä | `hdd_sq` (lämmitystarveluvun neliö, kynnys 17°C) | positiivinen |
| Tuulen epälineaarisuus | `wind_log_scarcity` = log1p(max(0, 8-tuuli)) | positiivinen (vähäinen tuuli = korkeampi hinta) |
| Tuuli × kysyntä | `wind_calm_x_peak_am` = max(0, 6-tuuli) × aamuhuippu (klo 9, σ=1,8) | positiivinen |
| Tuuli × kysyntä | `wind_calm_x_peak_pm` = max(0, 6-tuuli) × iltahuippu (klo 19, σ=2,0) | positiivinen |

### Taso 2: AR-naapurihinnat + vientipotentiaali (+4) — ei API-avaimia

AR(2)-mallit ennustavat rajat ylittäviä naapurihintoja käyttäen arkipäivä/viikonloppu-tuntiprofiileja ja vaimennettua autoregressiivista poikkeamaa.

| Piirre | Lähde | Menetelmä |
|--------|-------|-----------|
| `ar_se1` | Ruotsi SE1 | AR(2) poikkeamasta tuntityyppiprofiilista, normalisoitu ÷100 |
| `ar_se3` | Ruotsi SE3 | AR(2) poikkeamasta tuntityyppiprofiilista, normalisoitu ÷100 |
| `ar_ee` | Viro | AR(2) poikkeamasta tuntityyppiprofiilista, normalisoitu ÷100 |
| `export_potential_se3` | SE3-hintaero | max(0, -spread_7d_fi_se3) |

AR-poikkeama vaimennetaan (maksimijuuri < 0,95), joten ennusteet konvergoituvat päiväprofiiliin 24 tunnissa, mikä takaa vakauden koko 170 tunnin ennusteikkunassa.

### Taso 3: Ydinvoimapiirteet (+0-2) — vaatii Fingrid API-avaimen

| Piirre | Kaava | Merkitys |
|--------|-------|----------|
| `nuclear_deficit` | max(0, 1 - nuclear_mw/4372) | Ydinvoimakapasiteetin puuteosuus |
| `nuclear_x_scarcity` | nuclear_deficit × niukkuusindikaattori | Ydinvoimaseisokki vahvistaa sääpohjaista niukkuutta |

**Ennakollinen seisokkiraportointi:** Suunnitellut seisokkiaikataulut haetaan [Nord Pool UMM -alustalta](https://umm.nordpoolgroup.com/) (julkinen rajapinta, ei avainta). Koordinaattori laskee ydinvoiman saatavuuden tunneittain ennustehorisontissa.

### Piirteiden lukumäärä konfiguraation mukaan

| Konfiguraatio | Piirteet | API-avaimet |
|---------------|----------|-------------|
| Vain Taso 1 | 11 | Ei mitään |
| Taso 1 + 2 | 15 | Ei mitään |
| Taso 1 + 2 + 3 | 17 | 1 (Fingrid, ilmainen) |

---

## Malliarkkitehtuuri

### Tuntimalli: Log-lineaarinen Ridge-regressio

**Ennustekaava:** `hinta = skaala × max(0, exp(Σ kerroin_i × piirre_i + vakio) - 55) ^ potenssi`

Log-muunnos käsittelee luonnollisesti epälineaarisen hinta-niukkuussuhteen: lähes lineaarinen matalilla hinnoilla, eksponentiaalinen vahvistus korkeilla hinnoilla.

- Ridge-regressio kohteella log(hinta + 55)
- 17 merkkivalidoitua piirrettä
- Tehovenytys (skaala, eksponentti) sovitetaan Nelder-Mead-optimoinnilla
- Aikapainotus: puoliintumisaika 120 päivää
- Ridge alpha = 1,0, laajennettu matriisi (ei sakkoa vakiotermille)

**Koulutus:** 4 vuoden historiallinen data, aikajärjestetty 85/15 jako, eräkäsittely (512 riviä).

### Kestomalli: Segmenttihierarkkinen Ridge + PAVA

Ennustaa D(k) = keskimääräisen spot-hinnan halvimmille k tunnille päivässä. D(k) on matemaattisesti ekvivalentti ehdollisen riskin arvon (CVaR, Conditional Value-at-Risk) kanssa päivänsisäisestä hintajakaumasta tasolla α = k/24, mikä tekee siitä luonnollisen kustannusmittarin kuormien ajoitukseen: "ajoita halvimmille k tunnille" minimoi CVaR:n.

**PAVA** (Pool Adjacent Violators Algorithm) on isotonisen regression menetelmä, joka pakottaa monotonisuuden. Koska D(k) on määritelmän mukaan ei-vähenevä — useampien tuntien lisääminen keskiarvoon voi sisältää vain yhtä kalliita tai kalliimpia tunteja — PAVA yhdistää itsenäisten Ridge-ennusteiden rikkomukset keskiarvoistamalla vierekkäisiä pareja kunnes D(1) ≤ D(2) ≤ ... ≤ D(N) toteutuu kaikkialla.

**Arkkitehtuuri:**
- 4 päiväsegmenttiä: yö (22-05, 8h), aamu (06-09, 4h), keskipäivä (10-15, 6h), ilta (16-21, 6h)
- Jokainen (segmentti, kestotaso): itsenäinen Ridge-malli 10 piirteellä
- Log-lineaarinen kohde: log(D(k) + 55)
- Unohtamiskerroin λ = 0,960 (puoliintumisaika 17 päivää, optimoitu pyyhkäisyllä)
- PAVA isotoninen jälkikäsittely: pakottaa D(1) ≤ D(2) ≤ ... ≤ D(N)
- Segmentistä päivään -rekonstruktio: eristä lajitellut hinnat → yhdistä → uudelleenlajittelu → 24h D(k)

**Kestomallin piirteet:**
`wind_mean`, `solar_mean`, `hdd_mean`, `se3_mean`, `se1_mean`, `nuclear_deficit`, `is_workday`, `month_sin`, `month_cos`, `wind_log_scarcity`

**Suorituskyky (Spearmanin järjestyskorrelaatio):**

| Kestotaso | Käyttötapaus | ρ (kaikki) | ρ (viim. 365 pv) |
|:-:|:-:|:-:|:-:|
| D(1) | Halvin tunti | 0,895 | 0,898 |
| D(4) | Halvimmat 4h | 0,904 | 0,906 |
| D(8) | Halvimmat 8h | 0,929 | 0,921 |
| D(24) | Päivän keskiarvo | 0,935 | 0,937 |

**Tuloste:** `model_coefs.json` sisältäen tuntimallin kertoimet, AR-mallin parametrit ja kestomallin kertoimet.

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

## Kuluttajahinta ja ohjaussignaalit

**Kaava:** `(max(0, spot_EUR_MWh) / 1000 + marginaali + siirtohinta + energiavero) × ALV × 100` [c/kWh]

Konfiguroitavissa operaattorikohtaisesti tiedostossa `finland.yaml`. Oletus: Elenia (päivä 3,61, yö 2,20 c/kWh), ALV 25,5%, energiavero 2,325 c/kWh, myyjän marginaali 0,00 c/kWh (aseta sähkösopimuksesi mukaan).

**Sensorit:**

**Ennustesensorit (luodaan aina):**

| Sensori | Yksikkö | Kuvaus |
|---------|---------|--------|
| Spot Price Forecast | EUR/MWh | Ennustettu spot-hinta + 170h ennuste |
| Consumer Price | EUR/kWh | Kokonaishinta sis. marginaalin, siirtohinnan, ALV:n, energiaveron |
| Duration Forecast | c/kWh | D(k) kestokäyrät: päivittäinen kustannus käyttökeston mukaan (7 vrk ennuste) |
| Cheapest Hours | aikaleima | Halvimmat 1h/2h/3h/4h/6h/8h jaksot säädettävässä hakuikkunassa |
| Week Price Stats | EUR/kWh | Kuluttajahinnan min/keskiarvo/max ennusteikkunassa |

**Spot-hintasensorit (valinnainen, kun Nordpool-entiteetti on konfiguroitu):**

| Sensori | Yksikkö | Kuvaus |
|---------|---------|--------|
| Spot Electricity Price | EUR/kWh | Todellinen ostohinta Nordpoolista jatkuvalla aikajanalla |
| Spot Electricity Selling Price | EUR/kWh | Spot miinus aurinkosähkön myyntipalkkio |

**Duration Forecast** -sensori on ensisijainen kustannussuunnittelutyökalu. Sen tila on päivän D(4) kuluttajahinta c/kWh. Attribuutit sisältävät `today_d1`, `today_d4`, `today_d8`, `today_d24` avaintasoille sekä `daily_forecast` täydet 7 vrk D(k)-käyrät (24 tasoa per päivä, sekä kuluttaja-c/kWh että spot-EUR/MWh). Kaikki kuluttajahinnat käyttävät konfiguroituja tariffeja — ei kovakoodattuja arvoja.

**Cheapest Hours** -sensori tarjoaa alkamisajat ja keskihinnat halvimmille peräkkäisille N tunnin jaksoille säädettävässä hakuikkunassa sekä listan keskiarvon alittavista tunneista.

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

### Nykyinen suorituskyky (v2.0, 4 vuoden koulutusdata, 120 päivän puoliintumisaika)

**Tuntimalli:**

| Mittari | Arvo |
|---------|:---:|
| MAE | 23,6 EUR/MWh |
| RMSE | 47,1 EUR/MWh |
| R² | 0,522 |

**Kestomalli (Spearmanin ρ, viimeiset 365 päivää):**

| D(k) | Käyttötapaus | ρ |
|:---:|:-:|:---:|
| D(4) | Halvimmat 4h | 0,906 |
| D(8) | Halvimmat 8h | 0,921 |
| D(24) | Päivän keskiarvo | 0,937 |

### Suositeltu uudelleenkoulutustaajuus

**Kouluta uudelleen 3-4 kuukauden välein (neljännesvuosittain).**

### Uudelleenkoulutuksen suorittaminen

```bash
cd HA-spot-price-predictor
pip install -r requirements.txt

# Kouluta uusimmalla datalla
export FINGRID_API_KEY=avaimesi  # valinnainen
python -m src.train_model --region finland

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
├── tests/                       # 98 yksikkötestiä
└── output/                      # Tuotetut artefaktit
    ├── model_coefs.json
    ├── model_dashboard.html
    └── forecast.html
```
