# SlideDetection

Nástroj pro automatickou detekci slidů v přednáškových videích. Je součástí diplomové práce zaměřené na automatické zkrácení přednáškových videí pomocí pipeline STT → parafráze → TTS.

## Kontext — kam tento nástroj zapadá

Cílem diplomové práce je automaticky zkrátit přednáškové video (typicky 1 hodina → 30 minut) odstraněním filler slov (*um*, *em*, ...) a kompresí promluvy přes TTS. Dynamic Time Warping na takto velký časový rozdíl nestačí, proto se video nejprve rozseká na segmenty dle slidů. Každý segment pak prochází pipeline samostatně a výsledky se slepí zpět dohromady.

```
Video + audio  →  STT (přepis)  →  parafráze (komprese)  →  TTS (nové audio)
                                                                      ↑
                          SlideDetection rozdělí video na segmenty ──┘
                          a výsledné video = nové audio + obrázky slidů
```

## Co nástroj dělá

Ze vstupního videa:
1. Detekuje přechody mezi slidy analýzou rozdílů po sobě jdoucích snímků.
2. Čistí falešné přechody pomocí série post-processing kroků (mergování krátkých, identických a postupně-budovaných slidů).
3. Klasifikuje každý segment jako `slide`, `camera` nebo `demo`.
4. Spočítá **confidence skóre** každé hranice (kombinace vizuálního deltu a audio silence) a označí podezřelé hranice příznakem `needs_review`.
5. Uloží reprezentativní obrázek každého slidu (nejostřejší snímek z jeho stabilní části).
6. Spustí **OCR** (Tesseract) na exportovaných slidech a uloží extrahovaný text.
7. Vygeneruje `slides.json` s metadaty pro navazující pipeline a `detection.log` s plným záznamem zpracování.

### Typy segmentů a jejich dopad na pipeline

| Typ | Popis | Chování pipeline |
|---|---|---|
| `slide` | Normální prezentační slide | Parafráze s kompresí — TTS může být kratší |
| `camera` | Fullscreen kamera přednášejícího | 100 % zachování počtu slov — TTS musí odpovídat délce originálu |
| `demo` | Demo / IDE / terminál přes slide | 100 % zachování počtu slov |

## Výstup

```
detected_slides/
└── <název_videa>/
    ├── slide_001.jpg
    ├── slide_002.jpg
    ├── ...
    ├── slides.json
    └── detection.log
```

`slides.json` — příklad:
```json
[
  {
    "id": 1,
    "start": 0.0,
    "end": 76.52,
    "duration": 76.52,
    "type": "slide",
    "image": "slide_001.jpg",
    "text": "Úvod do strojového učení Hlavní typy úloh klasifikace regrese clustering",
    "confidence": 1.0,
    "needs_review": false
  },
  {
    "id": 2,
    "start": 76.52,
    "end": 94.12,
    "duration": 17.6,
    "type": "camera",
    "image": "slide_002.jpg",
    "text": "",
    "confidence": 0.54,
    "needs_review": true
  }
]
```

| Pole | Popis |
|---|---|
| `id` | Pořadové číslo slidu (1-based, sekvenční) |
| `start` / `end` | Časové hranice v sekundách |
| `duration` | `end - start` |
| `type` | `slide` / `camera` / `demo` |
| `image` | Soubor s reprezentativním snímkem |
| `text` | OCR text slidu (prázdný u kamery a demo segmentů, nebo když není Tesseract) |
| `confidence` | 0.0–1.0; míra jistoty že hranice na `start` je skutečný přechod |
| `needs_review` | `true` když `confidence < confidence_threshold` (default 0.6) |

## Instalace

```bash
pip install -r requirements.txt
```

**Povinné závislosti:** `opencv-python`, `numpy`, `tqdm`

**Volitelné (degradují gracefully, jen vypíšou warning):**
- **OCR** — `pip install pytesseract pillow` + systémový [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) v PATH + jazykový balíček `ces.traineddata` (česky) a `eng.traineddata` (anglicky). Bez nich pole `text` bude prázdné.
- **Audio cross-validace** — [ffmpeg](https://ffmpeg.org/) v PATH. Bez něj se audio skóre v confidence formuli nahradí neutrální hodnotou 0.5.

## Použití

```bash
python main.py [cesta_k_videu]
```

Nebo přímo v kódu:

```python
from main import SlideDetector

detector = SlideDetector(
    video_path="video.mp4",
    output_dir="detected_slides",
    threshold_percent=5,
    min_duration=10,
    similarity_threshold=5,
    min_changed_blocks=4,
    camera_segment_min_count=5,
    face_area_threshold=0.15,
    use_ocr=True,
    ocr_lang="ces+eng",
    use_audio_validation=True,
    confidence_threshold=0.6,
)
slides = detector.run()
```

---

## Problémy a jejich řešení

### 1. Výběr threshold pro detekci změny

**Problém:** Při threshold 20 % byly nalezeny jen 2 slidy (celé video jako jeden kus). Při threshold 2 % pomocí knihovny `scenedetect` bylo nalezeno až 37 segmentů ze stejného videa, kde se ve skutečnosti mění jen 12 slidů — komprese videa a pohyb kurzoru generovaly falešné přechody.

**Řešení:** Přechod na vlastní implementaci v OpenCV. Klíčová optimalizace: pixel, který se změní o méně než 25 (z 255), se ignoruje jako šum. Výsledkem je, že threshold 5 % s vlastní implementací dává stejný počet segmentů jako threshold 2 % u scenedetect (13 segmentů), ale bez falešných přechodů způsobených šumem.

Zvolená hodnota: **threshold 5 %, check každých 0,5 s** (2× za sekundu místo každého snímku — výrazně rychlejší).

---

### 2. Falešné přechody způsobené PiP kamerou v rohu

**Problém:** Přednášející má v rohu obrazovky malé okno s kamerou. Pohyb v tomto okně způsoboval, že detekce vyhodnotila přechod na nový slide, ačkoliv slide zůstal stejný.

**Řešení:** Analýza změn po blocích — frame se rozdělí do mřížky 4×4 (16 bloků). Slide-transition mění velkou část obrazovky (mnoho bloků), zatímco PiP kamera zabírá jen 1–2 bloky v rohu. Parametr `min_changed_blocks` (výchozí 4) filtruje změny v méně než 4 blocích.

---

### 3. Rychlé překliky (přednášející omylem přeskočí slide a vrátí se zpět)

**Problém:** Krátké segmenty (< `min_duration`) způsobené náhodným přepnutím slidu zpět a opět dopředu generovaly falešné slidy.

**Řešení:** Post-processing funkce `_merge_short_slides`:
- Načte reprezentativní snímek krátkého segmentu.
- Porovná ho se sousedními segmenty (předchozím i následujícím).
- Pokud předchozí a následující segment jsou si vizuálně podobné → presenter omylem přeskočil a vrátil se → sloučí všechny tři do jednoho.
- Pokud krátký segment je podobný předchozímu → sloučí do předchozího.
- Pokud je podobný následujícímu → sloučí do následujícího.
- Pokud je unikátní ale příliš krátký → sloučí dopředu (do následujícího), ne zpět.

---

### 4. Fullscreen kamera přednášejícího

**Problém:** Když přednášející mluví přímo do kamery (bez slidu), video obsahuje dlouhé kontinuální záběry tváře. Tyto záběry negenerovaly krátké segmenty, takže je metoda pro PiP kameru nezachytila.

**Řešení (tři vrstvy):**

**a) Detekce shluku krátkých segmentů** (`_merge_camera_segments`): Pokud se za sebou vyskytuje ≥ N krátkých segmentů (pohyb těla/rukou generuje drobné změny), celý shluk se sloučí do jednoho segmentu a označí jako `camera`.

**b) Haar Cascade detekce obličeje** (`_reclassify_by_face`): Z každého segmentu se odeberou 3 vzorkové snímky, na které se aplikuje Haar Cascade detekce obličeje (frontální + profilový). Pokud detekovaný obličej zabírá více než `face_area_threshold` (výchozí 15 %) plochy snímku, segment se překlasifikuje na `camera`. PiP okno v rohu má < 10 % plochy, takže neaktivuje překlasifikaci.

**c) Heuristika barevné saturace**: Fullscreen kamera má pleti a přirozené pozadí s vysokou saturací barev po celé ploše. Pokud > 40 % pixelů má saturaci > 60 (v HSV), segment se označí jako `camera`.

---

### 5. Falešné přechody způsobené jednorázovými artefakty (glitch)

**Problém:** Komprese videa nebo přechodová animace způsobila jeden nebo dva snímky, které vypadaly jako přechod — ale hned nato se obraz vrátil zpět.

**Řešení:** Přepínač `confirm_transitions` — po detekci potenciálního přechodu se načte snímek o jeden `check_interval` vpřed. Pokud se obraz vrátil k předchozímu stavu, přechod se zahodí jako glitch.

---

### 6. Vizuálně identické sousední slidy (false transition)

**Problém:** Drobné on-screen události (tooltip, kurzor, rychlá anotace) způsobily přechod mezi dvěma slidy, které jsou ve skutečnosti vizuálně identické.

**Řešení:** `_merge_similar_adjacent` — závěrečný průchod, který porovná reprezentativní snímky sousedních segmentů. Pokud je rozdíl menší než `similarity_threshold`, segmenty se sloučí.

---

### 7. Výběr nejlepšího reprezentativního snímku slidu

**Problém:** Snímek uprostřed segmentu mohl být rozmazaný (přechodová animace nebo pohyb řečníka).

**Řešení:** `_best_frame_at_slide` — z každého segmentu se odeberou 5 vzorkových snímků ze stabilní střední části (s vynecháním okrajů). Vrátí se nejostřejší snímek měřený Laplacianovou variancí.

---

### 8. Postupné odhalování bulletů (PowerPoint progressive builds)

**Problém:** PowerPoint a Keynote často odhalují bullety jeden po druhém. Každý reveal step překročí prahy detekce a vznikne tak série N "slidů", kde každý ukazuje neúplnou verzi téhož slidu. Pro pipeline to znamená rozsekané audio a thumbnaily neukazující finální obsah.

**Řešení:** `_merge_progressive_builds` — porovná dvojice sousedních `slide` segmentů a hledá tři podmínky současně:
- Horní ~55 % snímku je vizuálně identické (`change_pct < 1.5`) — nadpis a předchozí bullety se nezměnily.
- Dolní část se liší (`change_pct > 2.0`) — opravdu přibyl nový obsah, ne jen identický pár.
- Std pixelů v dolní části druhého snímku je vyšší než prvního (×1.10) — chrání proti opačnému případu, kdy byl obsah ubrán.

Při shodě se dvojice sloučí: drží se snímek pozdějšího slidu (nejúplnější verze) a `start` se posune zpět. Loop běží opakovaně a tak zvládne N-krokové buildy (např. 5 bulletů → 1 finální slide).

---

### 9. Nejistota detekce — confidence skórování + manuální review flag

**Problém:** Některé detekované hranice jsou jasné (velký vizuální skok, ticho v audiu), jiné jsou na hraně thresholdů. Bez kvantifikace si uživatel musel proklikat všechny slidy ručně.

**Řešení:** `_annotate_with_confidence` přidá ke každému slidu skóre 0.0–1.0 a flag `needs_review`. Skóre kombinuje:
- **Vizuální složku** — `0.5 · (change_pct / 20) + 0.5 · (changed_blocks / 16)`, clamped 0–1. Silné přechody saturují k 1.0.
- **Audio složku** — silence dip ratio v okně ±0.5 s kolem hranice (viz problém 10). Hluboké lokální minimum RMS = 0.9+, žádný dip = 0.2.
- **Kombinace** — `0.6 · visual + 0.4 · audio`. Vizuální signál má větší váhu, audio je doplňková validace.

Hranice s `combined < confidence_threshold` (default 0.6) dostanou `needs_review: true`. Downstream UI tak může uživatele rovnou navést na podezřelá místa.

---

### 10. Cross-validace přechodů přes audio (silence detection)

**Problém:** Vizuální detekce může reagovat na pohyb v PiP kameře, fade animaci nebo blesknutí kurzoru. Tyto falešné přechody nejsou doprovázeny řečovou pauzou, na rozdíl od skutečných přepnutí slidu, kdy řečník téměř vždy krátce ztichne.

**Řešení:** `_load_audio` jednorázově extrahuje mono PCM 16 kHz audio přes `ffmpeg` do dočasného WAV. Pak `_silence_score_at(t)`:
- Spočítá 50 ms RMS framy v okně ±0.5 s kolem `t`.
- Dip ratio = `min(rms) / mean(rms)`. Nízký poměr = výrazné lokální ticho.
- Skóre = `clip(1 - dip_ratio, 0.2, 1.0)`.

Skóre vstupuje do confidence (viz problém 9). Když ffmpeg chybí, audio score je 0.5 (neutrální) a confidence se počítá jen z vizuálního deltu.

---

### 11. Párování STT transkriptu se slidy (OCR)

**Problém:** Pipeline potřebuje vědět, které věty z STT transkriptu patří ke kterému slidu. Časové hranice z detektoru nejsou přesné na slovo — řečník může začít mluvit o dalším slidu pár sekund před přepnutím nebo dokončit větu o předchozím.

**Řešení:** `_ocr_slide_image` spustí Tesseract na každém exportovaném slidu (přes `pytesseract` + `PIL`) a uloží sjednocený jednořádkový text do `slide["text"]` a do JSONu. Downstream pipeline pak může fuzzy-matchovat úseky transkriptu na slidy podle obsahu, ne jen podle časových hranic.

OCR běží **až na finálních exportovaných obrázcích** (po všech merge fázích) — žádný overhead během detekce. Kamera a demo segmenty se vynechávají (zřídka obsahují čitelný text). Když Tesseract chybí, pole `text` je prázdné.

---

## Optimalizace rychlosti

Nástroj je navržen tak, aby zpracoval 1h video za jednotky minut na běžném CPU. Klíčové optimalizace:

### Zásadní (řádový speedup)

1. **`cv2.cap.grab()` místo `read()` pro přeskočené snímky** — `read()` plně dekóduje frame (drahá operace), `grab()` jen posune pozici. Hlavní smyčka volá `grab()` `step_frames - 1` krát a teprve poslední snímek dekóduje. Pro `check_interval=0.5s` při 30 FPS to znamená 14 `grab()` + 1 `read()` místo 15 `read()` — **~10× rychlejší** než plné čtení každého snímku.

2. **`check_interval=0.5s` (2× za sekundu)** — místo zkoumání každého snímku se vzorkuje 2× za sekundu. Slidy se nepřepínají rychleji než jednou za pár sekund, takže ztráta granularity je akceptovatelná.

3. **Gaussian blur + 25-pixel pixel-threshold** — preprocessing snižuje šum kompresního artefaktu. Díky tomu může vyšší `threshold_percent` (5 %) ignorovat šum bez zahození skutečných přechodů. Nižší prahy by vyžadovaly per-frame analýzu.

### Důležité

4. **4×4 grid block analysis místo per-pixel masks** — `_count_changed_blocks` rozdělí frame na 16 bloků a zkoumá jen agregát na blok. Spočítá se v µs i pro 4K snímky.

5. **Audio se extrahuje jednorázově na začátku** — `ffmpeg` proběhne jednou a vrátí celý PCM stream do paměti. Per-boundary `_silence_score_at` pak jen krájí numpy array (instant). Pro 1h videa = ~115 MB RAM (mono 16 kHz int16).

6. **OCR/face detection jen na finálních segmentech** — drahé operace běží `O(n_slides)` po post-processingu, ne `O(n_frames)` během detekce. Tesseract ~200 ms/slide × 50 slidů = 10 s celkem.

7. **Confidence skórování až po všech merge fázích** — boundary confidence se počítá z FINÁLNÍCH hranic, takže merge operace neovlivní cached metriky. Vyhne se nutnosti udržovat metriky napříč mutacemi seznamu.

### Drobné

8. **Lazy CascadeClassifier load** — Haar kaskády se načtou jen když je `use_face_detection=True`.

9. **Optional dependencies degradují gracefully** — chybějící `pytesseract` nebo `ffmpeg` jen vypnou příslušnou feature, ne hodí výjimku.

### Kde je prostor na další zrychlení

- **Multi-threaded OCR** — Tesseract je single-threaded a IO-bound. Pool 4 workerů by zkrátil OCR fázi na ~25 % současného času.
- **Streaming audio analýza** — místo načtení celého audia do RAM by se dalo `ffmpeg` pipovat a počítat RMS framy on-the-fly. Důležité pro >2h videa.
- **GPU akcelerace `_calculate_change_percentage`** — přes `cv2.cuda.absdiff` (vyžaduje OpenCV s CUDA buildem). Pro CPU stačí to co je dnes.

---

## Logging

Veškerý výstup se loguje přes Python `logging` modul:

- **`<output_dir>/detection.log`** — plný záznam každého běhu s timestampy a úrovní (`INFO` / `WARNING` / `ERROR`). Mód `w` — pro každý nový běh se přepíše.
- **stdout** — paralelně se vypisuje na konzoli (jen `%(message)s`, bez timestampů, ať to vypadá jako původní `print`).

Logger je modul-level `getLogger("SlideDetector")` a setup se volá v `_initialize_capture` (každý detektor instance má vlastní per-video log soubor). Při dávkovém zpracování více videí v `__main__` se starý FileHandler před připojením nového odpojí, takže logy mezi videi netečou.

`tqdm` progress bar zůstává na `stderr`, takže se s loggerem nemíchá.

---

## Architektura post-processingu (pořadí je důležité)

```
Raw detekce přechodů (s confirm_transitions filtrem proti glitchům)
        ↓
_merge_camera_segments        ← shluk ≥N krátkých segmentů = camera
        ↓
_merge_short_slides           ← krátké segmenty absorbuje do sousedů
        ↓
_reclassify_by_face           ← Haar Cascade + saturace → camera
        ↓
_merge_consecutive_noncontent ← sloučí sousední camera/camera nebo demo/demo
        ↓
_merge_progressive_builds     ← PowerPoint bullet-by-bullet → 1 slide
        ↓
_merge_similar_adjacent       ← vizuálně identické sousední slide → sloučit
        ↓
_annotate_with_confidence     ← spočítá confidence + needs_review per slide
        ↓
_export_slide_images          ← nejostřejší snímek + OCR text
        ↓
_export_json                  ← slides.json pro navazující pipeline
```

> Pořadí je kritické: `_merge_camera_segments` musí předcházet `_merge_short_slides`, jinak by krátkých segmentů bylo méně a camera-detection by je minula. `_annotate_with_confidence` musí běžet **až po** všech merge fázích, protože měří confidence z finálních hranic.

## Parametry SlideDetector

### Detekce přechodů

| Parametr | Default | Popis |
|---|---|---|
| `threshold_percent` | 1.0 | % změněných pixelů pro detekci přechodu |
| `min_duration` | 2.0 | Minimální délka segmentu (s) |
| `check_interval` | 0.5 | Interval vzorkování snímků (s); vyšší = rychlejší |
| `min_changed_blocks` | 4 | Min počet bloků (z 16) se změnou — filtruje PiP kameru |
| `confirm_transitions` | True | Potvrzení přechodu snímkem o interval vpřed (filtruje glitche) |

### Post-processing

| Parametr | Default | Popis |
|---|---|---|
| `similarity_threshold` | 2.0 | Max % rozdílu pro sloučení vizuálně identických slidů |
| `camera_segment_min_count` | 5 | Min počet krátkých segmentů v řadě pro označení jako camera |
| `use_face_detection` | True | Zapne/vypne Haar Cascade překlasifikaci na camera |
| `face_area_threshold` | 0.15 | Min poměr plochy obličeje k ploše snímku pro překlasifikaci |

### OCR + audio + confidence

| Parametr | Default | Popis |
|---|---|---|
| `use_ocr` | True | Zapne Tesseract OCR na exportovaných slidech |
| `ocr_lang` | `"ces+eng"` | Jazyky pro Tesseract; vyžaduje odpovídající `*.traineddata` |
| `use_audio_validation` | True | Zapne silence-based cross-validaci hranic přes ffmpeg |
| `audio_sr` | 16000 | Vzorkovací frekvence pro audio analýzu (Hz) |
| `confidence_threshold` | 0.6 | Hranice pod kterou je `needs_review = true` |

---

## TODO / další možná rozšíření

Vysoká hodnota:
- **ROI maska pro PiP kameru / titulky** — exponovat `roi_mask` (np.ndarray nebo bbox) co se aplikuje v `_calculate_change_percentage` i `_count_changed_blocks`. Robustnější než heuristika přes block count.
- **Detekce fade/dissolve přechodů** — pomalý cross-fade může generovat několik za sebou jdoucích "přechodů". Logika: po detekci přechodu se podívat na 3-5 framů dopředu a pokud `change_pct` klesá monotónně k 0, vzít až poslední ustálený frame.

Nice-to-have:
- **Slide deduplication napříč videem** — perceptual hash (`imagehash`) na exportovaných slidech, slidy s Hammingovou vzdáleností < N dostanou stejný `cluster_id`. Užitečné když se řečník vrací zpět.
- **HTML preview report** — `report.html` v `output_dir` s thumbnaily, časy, confidence, OCR textem a zvýrazněním `needs_review`. Šetří hodiny při ladění.
- **Per-video config presety** — YAML profily (`presets/zoom.yaml`, `presets/screencast.yaml`) místo hardcoded params v `__main__`.
- **OCR upscale** — `cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)` před Tesseractem výrazně zvedne přesnost na malém textu.
- **CLI argparse** — místo hardcoded listu v `__main__` přijímat `--video`, `--output-dir`, `--config` jako argumenty.

Code quality:
- **Refactor do modulů** — `main.py` má teď ~860 řádků. Rozdělit na `detector.py`, `postprocess.py`, `classify.py`, `confidence.py`, `audio.py`, `export.py`.
- **Regression test set** — 3-5 krátkých klipů s ručně anotovaným ground-truth + skript pro precision/recall přechodů.
