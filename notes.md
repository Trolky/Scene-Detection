# Výpisky k detekci slidů v obraze

## Požadavky na software 1 (z ZSWI) pomocí knihovny scenedetect
 - Output by mělo být 12 slidů, které se v prezentaci vyskytují.
 - Počet framů, které zpracovává je 57225.
### Theshold 20%

- Našlo to jenom 2 slidy. 
- Slide 1: 00:00:00.000 - 00:01:16.520
- Slide 2: 00:01:16.520 - 00:38:09.000

Z tohoto vyplívá, že je potřeba snížit threshold, aby se našly i další slidy.

### Theshold 5%

- Našlo 13 slidů
- Slide 1: 00:00:00.000 - 00:01:16.520 (Duration: 76.52s)
- Slide 2: 00:01:16.520 - 00:01:34.120 (Duration: 17.60s)
- Slide 3: 00:01:34.120 - 00:03:41.760 (Duration: 127.64s)
- Slide 4: 00:03:41.760 - 00:04:47.080 (Duration: 65.32s)
- Slide 5: 00:04:47.080 - 00:04:48.480 (Duration: 1.40s)
- Slide 6: 00:04:48.480 - 00:13:52.560 (Duration: 544.08s)
- Slide 7: 00:13:52.560 - 00:17:18.080 (Duration: 205.52s)
- Slide 8: 00:17:18.080 - 00:28:29.440 (Duration: 671.36s)
- Slide 9: 00:28:29.440 - 00:30:00.640 (Duration: 91.20s)
- Slide 10: 00:30:00.640 - 00:32:31.560 (Duration: 150.92s)
- Slide 11: 00:32:31.560 - 00:33:22.480 (Duration: 50.92s)
- Slide 12: 00:33:22.480 - 00:37:04.320 (Duration: 221.84s)
- Slide 13: 00:37:04.320 - 00:38:09.000 (Duration: 64.68s)

Slidy 4, 5 a 6 by měl být jako jeden slide. Slide 5 je rychlí překlik na slide 7 a hned zpět na původní.
A slide 6 je vlastně "pokračování" slidu 4.

Slide 8 by měl být rozdělen na dva slidy, protože se tam mění obsah. 
V čase 0:27:08 se přepíná na další slide, ale tyto slidy jsou velmi podobné, takže je potřeba snížit threshold ještě více, aby se našly i tyto slidy.

### Theshold 2%

- Našlo 37 slidů
- Slide 1: 00:00:00.000 - 00:01:16.520 (Duration: 76.52s)
- Slide 2: 00:01:16.520 - 00:01:34.120 (Duration: 17.60s)
- Slide 3: 00:01:34.120 - 00:01:44.120 (Duration: 10.00s)
- Slide 4: 00:01:44.120 - 00:02:14.120 (Duration: 30.00s)
- Slide 5: 00:02:14.120 - 00:02:54.120 (Duration: 40.00s)
- Slide 6: 00:02:54.120 - 00:03:24.120 (Duration: 30.00s)
- Slide 7: 00:03:24.120 - 00:03:41.760 (Duration: 17.64s)
- Slide 8: 00:03:41.760 - 00:03:51.760 (Duration: 10.00s)
- Slide 9: 00:03:51.760 - 00:04:01.760 (Duration: 10.00s)
- Slide 10: 00:04:01.760 - 00:04:47.080 (Duration: 45.32s)
- Slide 11: 00:04:47.080 - 00:04:48.480 (Duration: 1.40s)
- Slide 12: 00:04:48.480 - 00:04:58.480 (Duration: 10.00s)
- Slide 13: 00:04:58.480 - 00:05:08.480 (Duration: 10.00s)
- Slide 14: 00:05:08.480 - 00:13:52.560 (Duration: 524.08s)
- Slide 15: 00:13:52.560 - 00:13:58.480 (Duration: 5.92s)
- Slide 16: 00:13:58.480 - 00:14:08.480 (Duration: 10.00s)
- Slide 17: 00:14:08.480 - 00:14:38.480 (Duration: 30.00s)
- Slide 18: 00:14:38.480 - 00:14:48.480 (Duration: 10.00s)
- Slide 19: 00:14:48.480 - 00:15:28.480 (Duration: 40.00s)
- Slide 20: 00:15:28.480 - 00:15:58.480 (Duration: 30.00s)
- Slide 21: 00:15:58.480 - 00:16:18.480 (Duration: 20.00s)
- Slide 22: 00:16:18.480 - 00:16:38.480 (Duration: 20.00s)
- Slide 23: 00:16:38.480 - 00:17:18.080 (Duration: 39.60s)
- Slide 24: 00:17:18.080 - 00:27:08.320 (Duration: 590.24s)
- Slide 25: 00:27:08.320 - 00:28:29.440 (Duration: 81.12s)
- Slide 26: 00:28:29.440 - 00:30:00.640 (Duration: 91.20s)
- Slide 27: 00:30:00.640 - 00:30:10.640 (Duration: 10.00s)
- Slide 28: 00:30:10.640 - 00:30:20.640 (Duration: 10.00s)
- Slide 29: 00:30:20.640 - 00:31:00.640 (Duration: 40.00s)
- Slide 30: 00:31:00.640 - 00:32:31.560 (Duration: 90.92s)
- Slide 31: 00:32:31.560 - 00:32:40.640 (Duration: 9.08s)
- Slide 32: 00:32:40.640 - 00:32:44.360 (Duration: 3.72s)
- Slide 33: 00:32:44.360 - 00:32:50.640 (Duration: 6.28s)
- Slide 34: 00:32:50.640 - 00:33:00.640 (Duration: 10.00s)
- Slide 35: 00:33:00.640 - 00:33:20.640 (Duration: 20.00s)
- Slide 36: 00:33:20.640 - 00:33:22.480 (Duration: 1.84s)
- Slide 37: 00:33:22.480 - 00:37:04.320 (Duration: 221.84s)
- Slide 38: 00:37:04.320 - 00:38:09.000 (Duration: 64.68s)

Spousta slidů navíc i když se slide němění. Toto může způsobovat například kurzor (i když je neviditelný ve videu).
Komprese videa, která může způsobovat artefakty, které jsou detekovány jako změna slidů. Nebo jiný šum.

## Požadavky na software 1 (z ZSWI) pomocí knihovny OpenCV (vlastní implementace)
- Output by mělo být 12 slidů, které se v prezentaci vyskyt
- Počet framů, které zpracovává je 4768.
- Proto kontroluji slidy třeba jenom 2x, 3x za sekundu, aby se zrychlil výpočet.
- Vlastní implementace je pomalejší než použití knihovny scenedetect.
- Je potřeba optimalizace, že zmenším obraz na danou velikost, aby se zrychlil výpočet.

## Threshold 20%, min. délka slidu 2s a check interval 0.5 (2x za sekundu)

- Našlo 2 slidy
- Slide 1: 00:00 - 01:16 (Duration: 76.80s)
- Slide 2: 01:16 - 38:09 (Duration: 2212.20s)

## Threshold 5%, min. délka slidu 2s a check interval 0.5 (2x za sekundu)

- Našlo 13 slidů
- Slide 1: 00:00:00.000 - 00:01:16.799 (Duration: 76.80s)
- Slide 2: 00:01:16.799 - 00:01:34.560 (Duration: 17.76s)
- Slide 3: 00:01:34.560 - 00:03:41.759 (Duration: 127.20s)
- Slide 4: 00:03:41.759 - 00:04:47.519 (Duration: 65.76s)
- Slide 5: 00:04:47.519 - 00:13:52.799 (Duration: 545.28s)
- Slide 6: 00:13:52.799 - 00:17:18.240 (Duration: 205.44s)
- Slide 7: 00:17:18.240 - 00:27:08.640 (Duration: 590.40s)
- Slide 8: 00:27:08.640 - 00:28:29.759 (Duration: 81.12s)
- Slide 9: 00:28:29.759 - 00:30:00.960 (Duration: 91.20s)
- Slide 10: 00:30:00.960 - 00:32:31.680 (Duration: 150.72s)
- Slide 11: 00:32:31.680 - 00:33:22.559 (Duration: 50.88s)
- Slide 12: 00:33:22.559 - 00:37:04.320 (Duration: 221.76s)
- Slide 13: 00:37:04.320 - 00:38:09.000 (Duration: 64.68s)

Slide 4, 5 a 6 by měl být jako jeden slide. Slide 5 je rychlí překlik na slide 7 a hned zpět na původní. Může být opraveno zadání větší min. délky slidu.
Slide 8 oproti knihovně scenedetect je rozdělen na dva slidy, protože se tam mění obsah.

## Threshold 2%, min. délka slidu 2s a check interval 0.5 (2x za sekundu)

- Našlo 13 slidů - stejné jako u theshold 5%
- Slide 1: 00:00:00.000 - 00:01:16.799 (Duration: 76.80s)
- Slide 2: 00:01:16.799 - 00:01:34.560 (Duration: 17.76s)
- Slide 3: 00:01:34.560 - 00:03:41.759 (Duration: 127.20s)
- Slide 4: 00:03:41.759 - 00:04:47.519 (Duration: 65.76s)
- Slide 5: 00:04:47.519 - 00:13:52.799 (Duration: 545.28s)
- Slide 6: 00:13:52.799 - 00:17:18.240 (Duration: 205.44s)
- Slide 7: 00:17:18.240 - 00:27:08.640 (Duration: 590.40s)
- Slide 8: 00:27:08.640 - 00:28:29.759 (Duration: 81.12s)
- Slide 9: 00:28:29.759 - 00:30:00.960 (Duration: 91.20s)
- Slide 10: 00:30:00.960 - 00:32:31.680 (Duration: 150.72s)
- Slide 11: 00:32:31.680 - 00:33:22.559 (Duration: 50.88s)
- Slide 12: 00:33:22.559 - 00:37:04.320 (Duration: 221.76s)
- Slide 13: 00:37:04.320 - 00:38:09.000 (Duration: 64.68s)

Oproti knihovně scenedetect se nenašlo tolik slidů navíc. Může to být tím, že ve vlastní implemntaci pokud se pixel změnil jen o 25 (z 255) tento pixel ignorujeme.
Také to může být tím, že ve vlastní implementaci nezmenšujeme obraz.