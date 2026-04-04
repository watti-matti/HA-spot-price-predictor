# Tekninen toteutus: HA-spot-price-predictor

Suomen sähkön spot-hinnan ennustaminen Home Assistantiin Ridge-regressiolla käyttäen fysiikkapohjaisia piirteitä ja useiden datalähteiden integrointia.

## Arkkitehtuuri

Järjestelmä koostuu kahdesta vaiheesta: **koulutus** (Python, ajetaan ajoittain PC:llä) ja **päättely** (Home Assistant, jatkuvasti päällä).

![Arkkitehtuurikuva](docs/diagrams/architecture-overview.drawio.png)

*Lähde: [docs/diagrams/architecture-overview.drawio](docs/diagrams/architecture-overview.drawio)*

### Koulutusputki

```
Sahkotin API ──┐
Open-Meteo API ─┼──> Piirre-engineering ──> Kaksivaiheinen Ridge ──> model_coefs.json
mgrey.se API ───┤    (28-38 piirrettä)      Regressio
Elering API ────┤
Fingrid API ────┘ (valinnainen)
```

### Home Assistant -käyttöönotto

```
REST-sensorit ──> Painotettu keskiarvo ──> Spot-hintaennuste ──> Kuluttajahinta
(7-11 kpl)        Template-sensori         (Jinja2-päättely)     + Ohjaussignaalit
                                                                  + Kojelauta
```

![Tietovirta](docs/diagrams/data-flow.drawio.png)

*Lähde: [docs/diagrams/data-flow.drawio](docs/diagrams/data-flow.drawio)*

---

## Datalähteet

### Pakolliset (ilmaiset, ei tunnistautumista)

| Lähde | Tarkoitus | Pyyntöraja |
|-------|-----------|------------|
| [Sahkotin API](https://sahkotin.fi/prices) | Suomen Nord Pool spot-hinnat (EUR/MWh) | Rajoittamaton |
| [Open-Meteo API](https://api.open-meteo.com) | Tuuli (120m), aurinko (45° kallistus), lämpötila | 10 000/vrk |
| [Open-Meteo Archive](https://archive-api.open-meteo.com) | Historiallinen säädata koulutukseen | 10 000/vrk |

### Rajat ylittävät hintalähteet (ilmaiset, ei tunnistautumista)

| Lähde | Alueet | Tarkoitus |
|-------|--------|-----------|
| [mgrey.se](https://mgrey.se/espot/api) | SE1, SE3 | Ruotsin spot-hinnat hintaeron laskentaan |
| [Elering API](https://dashboard.elering.ee/api/nps/price) | EE | Viron spot-hinnat hintaeron laskentaan |

Käytetään 7 päivän liukuvan hintaeron laskentaan, josta johdetaan `import_potential_xx` / `export_potential_xx` -piirteet. Analyysissa vahvistettu vahva autokorrelaatio (viikkotason lag-1 r=0,54-0,73, suunnan pysyvyys 100%).

### Valinnainen verkkodata (ilmainen API-avain)

| Lähde | Tarkoitus |
|-------|-----------|
| [Fingrid Open Data](https://data.fingrid.fi) | Ydinvoimatuotanto (#188), siirtokapasiteetti SE1-FI (#24), SE3-FI (#27), EE-FI (#115) |

Rekisteröidy ilmaiseksi osoitteessa data.fingrid.fi. Ilman tätä avainta malli koulutetaan vain Taso 1+2 -piirteillä.

---

## Piirre-engineering

### Taso 1: Peruspiirteet (28) — ei API-avaimia tarvita

| Kategoria | Määrä | Piirteet |
|-----------|-------|----------|
| Tarjontapuoli | 3 | `wind_speed_weighted`, `solar_irradiance_weighted`, `temperature_weighted` |
| Aikajaksot | 4 | `hour_sin/cos`, `month_sin/cos` |
| Kysyntämallit | 8 | `double_peak_am/pm` (Gauss, keskipiste 9h/19h), viikonloppuvariantit, `sauna_hour`, `monday_ramp`, `is_holiday`, `is_weekend` |
| Lämpökysyntä | 6 | `hdd`, `hdd_sq`, `daylight_deficit`, ristitermit (`wind_x_hdd`, `solar_x_deficit`, `temp_x_hdd`) |
| Fysiikkakorjattu tarjonta | 3 | `wind_power_density` (tiheykorjattu), `solar_power_temp` (NOCT-malli), `renewable_surplus` |
| Niukkuus | 4 | `scarcity_indicator`, `wind_drought_penalty`, `cold_morning_stress`, `cold_calm_dark` (Dunkelflaute) |

### Taso 2: Rajat ylittävän kaupan piirteet (6) — ei API-avaimia tarvita

Johdettu 7 päivän liukuvasta hintaerosta:

| Piirre | Kaava | Merkitys |
|--------|-------|----------|
| `import_potential_se1` | max(0, spread_7d_fi_se1) | Hintakannustin tuoda SE1:stä |
| `import_potential_se3` | max(0, spread_7d_fi_se3) | Hintakannustin tuoda SE3:sta |
| `import_potential_ee` | max(0, spread_7d_fi_ee) | Hintakannustin tuoda Virosta |
| `export_potential_se1` | max(0, -spread_7d_fi_se1) | Hintakannustin viedä SE1:een |
| `export_potential_se3` | max(0, -spread_7d_fi_se3) | Hintakannustin viedä SE3:een |
| `export_potential_ee` | max(0, -spread_7d_fi_ee) | Hintakannustin viedä Viroon |

### Taso 3: Verkkoinfrastruktuuripiirteet (0-4) — vaatii Fingrid API-avaimen

| Piirre | Lähde | Normalisointi |
|--------|-------|---------------|
| `nuclear_mw` | Fingrid #188 | 0-1 (0-4372 MW) |
| `import_capacity_se1` | Fingrid #24 | 0-1 (0-1500 MW) |
| `import_capacity_se3` | Fingrid #27 | 0-1 (0-1200 MW) |
| `import_capacity_ee` | Fingrid #115 | 0-1 (0-1016 MW) |

### Piirteiden lukumäärä konfiguraation mukaan

| Konfiguraatio | Piirteet | API-avaimet |
|---------------|----------|-------------|
| Vain Taso 1 | 28 | Ei mitään |
| Taso 1 + 2 | 34 | Ei mitään |
| Taso 1 + 2 + 3 | 38 | 1 (Fingrid, ilmainen) |

---

## Malliarkkitehtuuri

### Kaksivaiheinen Ridge-regressio paloittaisella kalibroinnilla

**Vaihe 1 (perusmalli):**
- Lineaarinen polynomi (aste 1) 28-38 piirteellä
- Painotetut normaaliyhtälöt: beta = (X'WX + alpha*I)^(-1) X'Wy
- Aikapainotus eksponentiaalisella vaimenemisella: w(t) = exp(-ln2 * ikä_tunnit / (365 * 24))
- Ridge alpha = 1,0

**Vaihe 2 (paloittainen kalibrointi):**
- Laajennetut piirteet: vaiheen 1 ennuste + 3 ReLU-murtopistettä
  - pw_relu_20 = max(0, s1 - 20)
  - pw_relu_40 = max(0, s1 - 40)
  - pw_relu_120 = max(0, s1 - 120)
- Korjaa systemaattisen harhan eri hintatasoilla

**Koulutus:** 4 vuoden historiallinen data, aikajärjestetty 85/15 jako, eräkäsittely (512 riviä).

**Tuloste:** `model_coefs.json` sisältäen vaiheiden 1 ja 2 kertoimet, piirteiden nimet ja tasojen tiedot.

---

## Kuluttajahinta ja ohjaussignaalit

**Kaava:** `(spot_EUR_MWh / 1000 + siirtohinta + energiavero) × ALV`

Konfiguroitavissa operaattorikohtaisesti tiedostossa `finland.yaml`. Oletus: Elenia (päivä 5,60, yö 4,30 c/kWh), ALV 25,5%, energiavero 2,253 c/kWh.

**Tulosignaalit (170 tunnin listat):**

| Signaali | Alue | Käyttötarkoitus |
|----------|------|-----------------|
| `price_with_tariff_forecast` | EUR/kWh | Absoluuttinen kuluttajahinta |
| `power_control_factor_pm1` | [-1, +1] | Halvin(+1) ... kallein(-1) |
| `power_control_factor_0_1` | [0, 1] | ON/OFF-kynnysarvo-ohjaus |
| `power_control_windowed_average_N_largest_0_1` | [0, 1] | Tasoitettu liukuva ikkuna |
| `power_control_windowed_average_N_largest_pm1` | [-1, +1] | Tasoitettu, bipolaarinen |

`*_pm1`- ja `*_0_1`-ohjaussignaalit perustuvat [T3m3z/spotprices2ha](https://github.com/T3m3z/spotprices2ha)-projektin lähestymistapaan. Signaalit soveltuvat ohjattavien kuormien säätöön (sähköauton lataus, lämminvesivaraaja, lämpöpumput) sekä edullisimpien N tunnin jaksojen etsintään.

---

## Home Assistant -integraatio

### Pyhäpäivien ja arkipäivien tunnistus

Malli käyttää arkipäivä/pyhäpäivä-tietoa kysyntämallien valintaan (arkipäivän huiput vs viikonloppu/pyhäpäivä). Kaksi vaihtoehtoa on tuettu ja integraatio voi vaihtaa niiden välillä `holidays.ha_workday_integration` -asetuksella aluekonfiguraatiossa.

#### Vaihtoehto A: HA:n Workday-integraatio (suositeltu)

Käyttää Home Assistantin sisäänrakennettua [Workday-integraatiota](https://www.home-assistant.io/integrations/workday/), joka tunnistaa pyhäpäivät automaattisesti yhteisön ylläpitämistä kalentereista.

**Asennus:**
1. **Asetukset** → **Laitteet ja palvelut** → **Lisää integraatio** → hae **Workday**
2. Aseta **Maa** arvoon `FI` (Suomi)
3. Jätä oletusasetukset (sulkee pyhäpäivät ja viikonloput arkipäivistä)
4. Integraatio luo `binary_sensor.workday_sensor` — **on** = arkipäivä, **off** = pyhäpäivä/viikonloppu

**Miten ennustaja käyttää sitä:**
- Koordinaattori kutsuu `workday.check_date` -palvelua jokaiselle tunnille 170h ennusteikkunassa
- Tämä palauttaa kunkin tulevan päivän arkipäivätiedon huomioiden kaikki Suomen pyhäpäivät
- Ei ylläpidettäviä pyhäpäivälistoja — Workday-integraatio päivittyy automaattisesti HA-päivitysten myötä

**Vaihto A-vaihtoehtoon:** Aseta `holidays.ha_workday_integration: true` tiedostossa `finland.yaml`. Tämä on oletusasetus.

#### Vaihtoehto B: Sisäänrakennettu pyhäpäivälaskuri

Käyttää aluekonfiguraatiotiedostossa (`finland.yaml`) määriteltyjä pyhäpäiväsääntöjä sisäänrakennetulla pääsiäisalgoritmilla.

**Milloin käyttää B-vaihtoehtoa:**
- Mallin testaus Home Assistantin ulkopuolella (koulutusputki käyttää aina B-vaihtoehtoa)
- Home Assistant ilman Workday-integraatiota
- Käyttöönotto alueelle, jota Workday-integraation pyhäpäiväkirjasto ei vielä tue

**Vaihto B-vaihtoehtoon:** Aseta `holidays.ha_workday_integration: false` tiedostossa `finland.yaml`.

### Sensorit tasoittain

Kaikki sensorit luodaan automaattisesti koulutetun mallin aktiivisten tasojen perusteella.

**Taso 1 (aina mukana):** 7 Open-Meteo REST-sensoria, painotettu keskiarvo, spot-hintaennuste, kuluttajahinta.

**Taso 2 (jos aktiivinen):** 3 rajat ylittävää hinta-REST-sensoria (mgrey.se, Elering), hintaeron laskenta.

**Taso 3 (jos aktiivinen):** 4 Fingrid REST-sensoria (ydinvoima, SE1, SE3, EE kapasiteetti).

---

## Alueellinen lokalisointi

Järjestelmää ohjaa yksittäinen aluekonfiguraatiotiedosto (`config/regions/finland.yaml`). Uuden alueen tukeminen:

1. **Tunnista säämittauspisteet** — etsi 5-8 maantieteellistä sijaintia kohdemaasta, jotka edustavat tuulivoiman, aurinkoenergian ja energiankulutuksen keskittymiä. Sijainnit painotetaan asennetun kapasiteetin (tuuli, aurinko) ja väestötiheyden (lämpötila/kysyntä) mukaan. Käytä alla olevaa tekoälykehotepohjaa.
2. **Luo uusi YAML-tiedosto** (esim. `sweden.yaml`) tunnistetuilla sijainneilla, painoilla ja paikallisilla parametreilla
3. **Määrittele paikallinen hinta-API** — etsi ilmainen rajapinta, joka tarjoaa day-ahead spot-hinnat kohteen tarjousalueelle
4. **Konfiguroi pyhäpäivät** — kiinteät päivämäärät, pääsiäiseen sidotut päivät ja maakohtaiset erikoissäännöt
5. **Lisää naapurimaiden hintalähteet** rajat ylittäville piirteille
6. **Aseta kuluttajahinnoittelu** — ALV-kanta, energiavero ja jakeluverkko-operaattorien tariffit
7. **Aja koulutus** komennolla `--region sweden`

Valinnaiset datalähteet ohitetaan automaattisesti, jos niiden API-avain puuttuu tai aluekonfiguraatio ei sisällä niitä.

Katso englanninkielisestä dokumentaatiosta ([TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md#finding-weather-locations-for-a-new-region)) tekoälykehotepohja uusien alueiden säämittauspisteiden tunnistamiseen.

---

## Tarkkuustavoitteet

| Konfiguraatio | Odotettu MAE | Huomautukset |
|---------------|-------------|--------------|
| Taso 1 (28 piirrettä) | ~29-30 EUR/MWh | Vastaa v3-perustasoa |
| Taso 1+2 (34 piirrettä) | ~25-28 EUR/MWh | Rajat ylittävät hintaerot auttavat |
| Taso 1+2+3 (38 piirrettä) | ~22-26 EUR/MWh | Ydinvoima + kapasiteetti ääriarvoille |

Arvioinnissa käytetään aikajärjestettyä 85/15 jakoa tunti-, kuukausi- ja segmenttitason (huippu/ei-huippu, arkipäivä/viikonloppu) erittelyllä.

---

## Projektin rakenne

```
HA-spot-price-predictor/
├── README.md
├── TECHNICAL_GUIDE.md           # Englanninkielinen dokumentaatio
├── TEKNINEN_TOTEUTUS.md         # Tämä dokumentti (suomeksi)
├── docs/diagrams/               # draw.io-arkkitehtuurikaaviot
├── requirements.txt
├── .env.example                 # FINGRID_API_KEY-paikkamerkki
├── .gitignore
├── custom_components/
│   └── spot_price_predictor/    # HACS-integraatio
├── src/
│   ├── train_model.py           # Koulutusputken pääohjelma
│   ├── features.py              # Dynaaminen piirre-engineering
│   ├── data_sources.py          # API-asiakasohjelmat (konfiguraatio-ohjattu)
│   ├── holidays.py              # Pyhäpäivälaskuri
│   └── evaluate.py              # Tarkkuusmittarit + visualisointi
├── config/regions/
│   └── finland.yaml             # Suomen aluekonfiguraatio
└── output/                      # Mallin tuotokset
```
