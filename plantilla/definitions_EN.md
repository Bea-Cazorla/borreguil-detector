# Field data for the wet-meadow app — definitions

> **DRAFT — to be reviewed by Beatriz P. Cazorla before sending.** The items marked
> **[TO DECIDE]** need a decision by the project. Partners should not use this
> version.

All partners use the same definitions, so that maps and moisture values can be
compared between parks. When in doubt about a point, mark it `doubtful = 1` rather
than guessing.

## The point

- `id`: any name or code for the point (optional; if missing, the app creates one).
- **One row per point**, at the place you observed. In a CSV, coordinates in decimal
  degrees (`lon`, `lat`). A layer (Shapefile, GeoPackage, GeoJSON, KML) can be in any
  projection as long as it has one (a Shapefile, with its .prj): the app converts it to
  WGS84 on its own.
- The app works with 10 × 10 m Sentinel-2 pixels: describe what is in the **10 m
  pixel around the point**, not the whole meadow.

## presence

- **1 — wet meadow.** A high-mountain wet meadow (in Sierra Nevada, *borreguil*:
  hygrophilous *Nardus* grassland / peaty meadow) covers the pixel around the
  point. **[TO DECIDE]**: minimum cover of meadow in the pixel to count as 1
  (proposal: half of the pixel or more).
- **0 — no wet meadow.** Anything else: dry grassland, scrub, rock, scree, bare
  soil, water. Absences are as important as presences: try to record some in the
  same areas, especially dry grasslands that look green.
- Leave empty if you did not look at presence: the app then only classifies the
  point; it is not used to check or train the model.

## environment (only if presence = 1)

| Value | Definition |
|---|---|
| `stream` | Along a watercourse, permanent or seasonal: the meadow follows the stream bed or its banks. **[TO DECIDE]** proposal: within about 30 m of the channel (the app proposes `stream` within 30 m of an OpenStreetMap stream) |
| `lake` | On the shore of a lake or pond, or in a basin that holds water part of the year. **[TO DECIDE]** proposal: within about 100 m of the water (the app proposes `lake` within 120 m of a known lake) |
| `slope` | Neither of the above: on a slope, fed by seepage, springs or snowmelt |

## moisture (only if presence = 1)

| Value | Definition |
|---|---|
| `wet` | The soil is **saturated** in mid-summer: water appears at the surface when you step on it, or there is shallow standing water |
| `dry` | The soil is **not saturated** in mid-summer, even if the meadow is green |

**[TO DECIDE]**:
- **When.** Moisture changes during the summer, so the observation date matters.
  Proposal: observe between **30 and 60 days after the snow has gone** from the
  site (not on a fixed calendar date, because snowmelt differs between mountains).
  Always record the `date`.
- **How.** Is a visual/tactile criterion enough, or do partners with a soil-moisture
  probe also record the measurement (in `notes`, or a new column)?
- **From imagery.** If moisture is judged on the orthophoto (`method =
  orthophoto`), which cues count as `wet` (open water, very dark green in late
  summer…)?

## purity (only if presence = 1)

What is inside the 10 m pixel:

| Value | Definition |
|---|---|
| `pure` | Almost all meadow (about 75 % or more) |
| `mixed-water` | Meadow with open water inside the pixel (pond, stream). The app proposes it when the pixel is water or touches water |
| `mixed-rock` | Meadow with rock, scree or bare soil inside the pixel |

## doubtful, method, date, observer, expertise, certainty, notes

- `doubtful = 1`: you are not sure about presence. Doubtful points are not used to
  train the model.
- `method`: `field` if the point was visited; `orthophoto` if it was judged on
  aerial imagery. Points reviewed inside the app are recorded as `orthophoto`;
  points added in the app by clicking on the map, by someone who knows the site,
  as `local knowledge`.
- `date`: date of the observation (`YYYY-MM-DD`).
- `observer`: who made the observation or the review (optional).
- `expertise`: profile of the observer: `student`, `manager` or `researcher`
  (optional).
- `certainty`: how sure the observer is about `presence` (optional): `high` (no
  doubt), `medium` (some doubt) or `low` (the most likely answer, but with quite a lot
  of doubt). If you cannot tell at all, use `doubtful = 1` instead: doubtful points
  are not used.
- `notes`: optional.

## What is optional

Only `lon`, `lat` and `presence` are required for the points that check or train the
model. The type columns are independent of
each other: a point with only `moisture` is useful for the moisture model. Fill in
what you know and leave the rest empty.
