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

Labels (YOLO format, class 0 = `gas_bottle`):

- `data/commons/labels/`: boxes for the 359 LPG bottle photos, made with our model A and then checked by us
  (model-assisted labeling, see the report section 3.4)
- `data/commons/label_review.csv`: result of the check per photo: `ok` (245), `missing bottles` (81), `wrong boxes` (33).
  The last 2 groups still have to be corrected by hand (upload `images/` + `labels/` + `data.yaml` to Roboflow or CVAT).
- `data/commons/train.txt`: the 245 checked photos, used for training model B

`data/commons/attribution.csv` has for every image: title, link, **author, license**, the Commons category and our
decision (`folder` + `review_note`). Most licenses (CC BY / CC BY-SA) require that we credit the author, so we have to
refer to this list in the report.

## 2. Labeled datasets on Roboflow Universe (download yourself)

Not in git because they're too big. How to download:

1. Open the link and sign in (free Roboflow account)
2. **Download Dataset** -> format **YOLOv11** (or YOLOv8, same format) -> "Download zip to computer"
3. Unzip into `data/roboflow/<name>/` with name `cylinders`, `gasoknok`, `gas` or `juntar` (ignored by git)

Or with the Roboflow API: put your private API key in a file `.env` in the root of the repo
(`ROBOFLOW_API_KEY=...`, never commit it) and run `python scripts/download_roboflow.py`.
(Downloading the zip link with curl / a script doesn't work: Roboflow puts a Cloudflare bot check in front of it.)

Then prepare them (all classes become one class `gas_bottle`):

```
python scripts/clean_export.py data/roboflow/cylinders    # same for gasoknok, gas, juntar
python scripts/prepare_primagaz.py                         # gasoknok + Gas -> data/datasets/primagaz
python scripts/to_one_class.py data/roboflow/juntar data/datasets/juntar
```

`clean_export.py` fixes 2 problems we had: some CylinDeRS images have names of 150+ characters, and together with our
OneDrive path that's more than the 260 characters Windows allows, so Ultralytics skipped them ("No such file or
directory"). And 3 CylinDeRS label files (+1 in gasoknok) mix polygons and boxes, which Ultralytics also skips.

| Dataset | Link | Notes |
|---|---|---|
| CylinDeRS | https://universe.roboflow.com/klearchos-stavrothanasopoulos-konstantinos-gkountakos-6jwgj/cylinders-iaq6n | 7,060 images, 25,269 boxes, split 70/20/10 (4,915 train / 1,434 val / 711 test). Mostly industrial cylinders (oxygen, welding...) but also steel and composite LPG bottles, stacks, cylinders on trucks, fallen cylinders. From a research paper ([Sensors 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11859601/)): YOLO11m got **0.91 mAP** on their test set (precision 0.905, recall 0.816) after 200 epochs. Good dataset to train a first model on. |
| gasoknok | https://universe.roboflow.com/gas-by6ce/gasoknok-vzwlk | 493 close-ups of real Primagaz bottles ("PRIMAGAZ", "UN1965", tare weights readable), filenames like `14_55_back_left_id_5_468.jpg` suggest several cameras filming the same bottles. Labels: `gas_ok` / `gas_nok`. |
| Gas | https://universe.roboflow.com/gas-by6ce/gas-vm7vi | 30 images: grey Primagaz bottles with the blue logo + dirty / taped bottles. Labels: `gas bottle`, `damaged bottle`, `wrong bottle`. Same workspace as gasoknok. |
| Juntar | https://universe.roboflow.com/testgasbottles/juntar | 393 overhead CCTV images (1920x1080) of trucks loaded with LPG bottles (Brazilian LPG distributor), 42,739 boxes, up to 270 bottles in one image (330 train / 37 val / 26 test). 11 classes: `p13`/`p20`/`p45` (bottle size) x colour. Interesting for a "count the bottles on a truck" setup. |

**Watch out with gasoknok:** the file names contain the bottle id (e.g. `14_32_id_143_back_left_1`), and in the
Roboflow split 50 bottles appear in more than one split (the same bottle in train and test). Test scores on that
split are too optimistic. `prepare_primagaz.py` splits again by bottle id: 247 train / 92 val / 184 test images
(the 30 `Gas` images all go to test).

Not checked in detail yet: [lpg_cylinder_green](https://universe.roboflow.com/demp/lpg_cylinder_green) (~1,000 images),
LPG Detection [2](https://universe.roboflow.com/coronafloat/lpg-detection-2) / [3](https://universe.roboflow.com/coronafloat/lpg-detection-3),
[Cyldet](https://universe.roboflow.com/final-dat/cyldet) (39 messy classes).

Every dataset uses different class names (`gas_cylinder`, `gas_ok`, `p13-blue`, ...). When we combine them we have to
map them all to our own class `gas_bottle` (class 0).

## 3. Videos (for tracking)

- The two example videos from the lecturer are from **MAKEEN Energy**, a Danish company that builds LPG filling plants
  (formerly Kosan Crisplant, their logo is in the videos):
  - Portugal: Petrogal LPG Cylinder Filling Plant (2013): https://youtu.be/UH-u0g3B8Pw
  - Bolivia: FLEXSPEED for Side-Entry Valves (2014): https://youtu.be/PtKnRpGvPM4
  - their channel has more filling plant videos: https://www.youtube.com/@MAKEENEnergy
- See `TODO` for downloading clips (yt-dlp) and extracting frames (ffmpeg). Keep clips short. `data/videos/` and
  `data/frames/` are not in git.
- Clips we use (cut with `scripts/cut_clip.py`, times in the original video):

| clip | from | what | camera |
|---|---|---|---|
| `petrogal_entry` | UH-u0g3B8Pw 2:28-2:48 | bottles enter the filling hall | moves, scene changes |
| `petrogal_worker` | UH-u0g3B8Pw 3:27-3:47 | worker checks bottles (seen from above) | moves |
| `petrogal_scale` | UH-u0g3B8Pw 4:28-4:48 | filling carousel + check-weigher | scene changes |
| `petrogal_checkweigher` | UH-u0g3B8Pw 4:42-4:54.5 | bottles pass the check-weigher, **our counting test** (4 bottles cross the line at x = 44 %) | **fixed** |
| `flexspeed_carousel` | PtKnRpGvPM4 0:18-0:33 | bottles on the filling carousel | moves |
| `flexspeed_workers` | PtKnRpGvPM4 1:13-1:28 | 2 workers walk by (tracking people) | moves |
| `flexspeed_conveyor` | PtKnRpGvPM4 2:04-2:24 | long conveyors full of bottles | moves |

## 4. Our own data (most valuable)

None of the data above really shows *our* situation (Primagaz bottles, in a Primagaz plant / depot). Ideas:

- photos / short videos of Primagaz bottles at shops or petrol stations (ask permission first)
- our own BBQ / heater bottle at home: different angles, lighting, backgrounds, partly hidden, several bottles together
- other brands too (useful later for "wrong bottle")

## 5. Licenses: can we use this data?

Yes, everything we train on is free to use (also commercially), as long as we credit the source:

- **Roboflow datasets** (CylinDeRS, gasoknok, Gas, Juntar): CC BY 4.0 -> credit the dataset (see the report, section 13).
- **Commons photos**: CC BY, CC BY-SA, CC0, public domain or GODL-India, per photo in `attribution.csv`. CC BY and
  CC BY-SA: credit the author. CC BY-SA ("share alike"): if we share changed versions of the photos, they must have the
  same license.
- **MAKEEN Energy videos**: normal YouTube videos, *not* free to share. We only use them locally to test tracking
  (not for training) and don't put them in the repo.
- **YOLO11 (Ultralytics)**: AGPL-3.0, fine for a school project. A closed commercial system (e.g. for Primagaz) needs an
  Ultralytics enterprise license.

## 6. Things to keep in mind (also for the report)

- **Domain gap**: a model trained on CylinDeRS / Commons sees mostly other bottles in other places. Its score on
  CylinDeRS doesn't tell us how well it works on Primagaz bottles, so we need test images that look like the real
  situation.
- **Split by source / video**, not by random frame (see `TODO`), otherwise the validation scores look too good.
- **Licenses**: credit the Commons authors (`attribution.csv`) and cite the Roboflow datasets we use.
- **Question for the lecturer**: the `gas` workspace (gasoknok + Gas) looks like an earlier group's Primagaz project.
  Can we reuse their images?
