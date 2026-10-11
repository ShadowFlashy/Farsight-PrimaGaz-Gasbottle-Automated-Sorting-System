# Data for gas bottle detection

Overview of the data we found / collected for detecting gas bottles. Summary:

| Source | What | Images | Labels | License | In this repo? |
|---|---|---|---|---|---|
| Wikimedia Commons | photos of LPG bottles (to label ourselves) | 359 (+102 other) | none yet | CC BY / BY-SA / CC0 / PD | yes, `data/commons/` |
| CylinDeRS (Roboflow) | research dataset, all kinds of gas cylinders | 7,060 | 1 class `gas_cylinder` | CC BY 4.0 | no, download (see below) |
| gasoknok (Roboflow) | close-ups of real **Primagaz** bottles | 493 | `gas_ok` / `gas_nok` | CC BY 4.0 | no, download |
| Gas (Roboflow) | Primagaz bottles, dirty / taped bottles | 30 | `gas bottle` / `damaged bottle` / `wrong bottle` | CC BY 4.0 | no, download |
| Juntar (Roboflow) | overhead camera of trucks full of LPG bottles | 393 | 11 classes (bottle size + colour) | CC BY 4.0 | no, download |
| MAKEEN Energy (YouTube) | videos of LPG filling plants (for tracking) | - | - | YouTube | no |

## 1. Wikimedia Commons photos (`data/commons/`)

Freely licensed photos from Wikimedia Commons, downloaded with `scripts/download_commons.py` from the LPG / propane /
butane cylinder categories (+ brand and country subcategories). Resized to max 1280 px wide.

We went through all 599 downloaded photos by hand and sorted them:

- `data/commons/images/` (359): **LPG bottles**, the kind of bottle Primagaz sells. Lots of variety: single bottles,
  bottle cages at shops / petrol stations, bottles on trucks, motorbikes and carts, depots, a few filling plant photos,
  close-ups of valves, different colours and brands (Butagaz, Repsol, Calor, Campingaz, Primagaz (c2247910), ...).
  These are the images to **label ourselves** in CVAT / Roboflow (class `gas_bottle`).
- `data/commons/other/` (102): other gas containers: industrial / medical / diving cylinders and camping cartridges.
  They don't look like Primagaz bottles. Could be used later as extra data or to test for false positives.
- `data/commons/rejected/` (138, **not in git**): no bottle visible or too small, drawings, explosion remains,
  big fixed tanks, near-duplicates.

`data/commons/attribution.csv` has for every image: title, link, **author, license**, the Commons category and our
decision (`folder` + `review_note`). Most licenses (CC BY / CC BY-SA) require that we credit the author, so we have to
refer to this list in the report.

## 2. Labeled datasets on Roboflow Universe (download yourself)

Not in git because they're too big. How to download:

1. Open the link and sign in (free Roboflow account)
2. **Download Dataset** -> format **YOLOv11** (or YOLOv8, same format) -> download zip
3. Unzip into `data/roboflow/<name>/` (this folder is ignored by git)

| Dataset | Link | Notes |
|---|---|---|
| CylinDeRS | https://universe.roboflow.com/klearchos-stavrothanasopoulos-konstantinos-gkountakos-6jwgj/cylinders-iaq6n | 7,060 images, 25,269 boxes, split 70/20/10 (4,915 train / 1,434 val / 711 test). Mostly industrial cylinders (oxygen, welding...) but also steel and composite LPG bottles, stacks, cylinders on trucks, fallen cylinders. From a research paper ([Sensors 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11859601/)): YOLO11m got **0.91 mAP** on their test set (precision 0.905, recall 0.816) after 200 epochs. Good dataset to train a first model on. |
| gasoknok | https://universe.roboflow.com/gas-by6ce/gasoknok-vzwlk | 493 close-ups of real Primagaz bottles ("PRIMAGAZ", "UN1965", tare weights readable), filenames like `14_55_back_left_id_5_468.jpg` suggest several cameras filming the same bottles. Labels: `gas_ok` / `gas_nok`. |
| Gas | https://universe.roboflow.com/gas-by6ce/gas-vm7vi | 30 images: grey Primagaz bottles with the blue logo + dirty / taped bottles. Labels: `gas bottle`, `damaged bottle`, `wrong bottle`. Same workspace as gasoknok. |
| Juntar | https://universe.roboflow.com/testgasbottles/juntar | 393 overhead CCTV images of trucks loaded with LPG bottles (Brazilian LPG distributor). 11 classes: `p13`/`p20`/`p45` (bottle size) x colour. Interesting for a "count the bottles on a truck" setup. |

Not checked in detail yet: [lpg_cylinder_green](https://universe.roboflow.com/demp/lpg_cylinder_green) (~1,000 images),
LPG Detection [2](https://universe.roboflow.com/coronafloat/lpg-detection-2) / [3](https://universe.roboflow.com/coronafloat/lpg-detection-3),
[Cyldet](https://universe.roboflow.com/final-dat/cyldet) (39 messy classes).

Every dataset uses different class names (`gas_cylinder`, `gas_ok`, `p13-blue`, ...). When we combine them we have to
map them all to our own class `gas_bottle` (class 0).

## 3. Videos (for tracking)

- The two example videos from the lecturer are from **MAKEEN Energy**, a company that builds LPG filling plants (part of
  SHV Energy, the same group as Primagaz):
  - Portugal: Petrogal LPG Cylinder Filling Plant (2013): https://youtu.be/UH-u0g3B8Pw
  - Bolivia: FLEXSPEED for Side-Entry Valves (2014): https://youtu.be/PtKnRpGvPM4
  - their channel has more filling plant videos: https://www.youtube.com/@MAKEENEnergy
- See `TODO` for downloading clips (yt-dlp) and extracting frames (ffmpeg). Keep clips short. `data/videos/` and
  `data/frames/` are not in git.

## 4. Our own data (most valuable)

None of the data above really shows *our* situation (Primagaz bottles, in a Primagaz plant / depot). Ideas:

- photos / short videos of Primagaz bottles at shops or petrol stations (ask permission first)
- our own BBQ / heater bottle at home: different angles, lighting, backgrounds, partly hidden, several bottles together
- other brands too (useful later for "wrong bottle")

## 5. Things to keep in mind (also for the report)

- **Domain gap**: a model trained on CylinDeRS / Commons sees mostly other bottles in other places. Its score on
  CylinDeRS doesn't tell us how well it works on Primagaz bottles, so we need test images that look like the real
  situation.
- **Split by source / video**, not by random frame (see `TODO`), otherwise the validation scores look too good.
- **Licenses**: credit the Commons authors (`attribution.csv`) and cite the Roboflow datasets we use.
- **Question for the lecturer**: the `gas` workspace (gasoknok + Gas) looks like an earlier group's Primagaz project.
  Can we reuse their images?
