# How the Lunar and Mars Crater Project Works

## 1. The project in one sentence

This project turns orbital images of the Moon and Mars into a visible chain of evidence: it finds possible craters, measures their image geometry, estimates a few physical descriptors, and uses the detected craters to demonstrate a possible landing point and rover route.

The goal is not to claim that a classroom-scale detector can safely guide a real spacecraft. The goal is to make the steps understandable and inspectable. A viewer can see the source image, intermediate image-processing results, detected outlines, measurements, assumptions, and the route that follows from those detections.

## 2. Why look for craters?

Impact craters are among the clearest repeating landforms in planetary images. Their rims and interiors can help scientists describe a surface, compare terrain, and identify places that a vehicle might need to avoid. Near a landing site, a crater is also a useful example of a hazard: the hollow interior, steep rim, and surrounding ejecta may all be undesirable for a lander or rover.

Finding a circle on an image is only the first step. A useful analysis should also answer questions such as:

- Where is the candidate crater, and how large does it appear in the image?
- What physical diameter does that pixel size imply at the tile's resolution?
- Does the rim look sharp or subdued under the image-processing measures used here?
- How much of the detected rim is supported by image edges?
- If these detected craters are treated as obstacles, where is a relatively clear touchdown point and how could a rover travel around them?

This application makes those steps visible in one place. The output can serve as a starting layer for GIS analysis, manual geological review, dataset quality checks, or a more capable navigation system. It is not a substitute for terrain elevation, rock maps, photogrammetry, or expert validation.

## 3. How the project grew from the original Moon version

The first version focused on lunar crater detection in Chandrayaan-2 OHRC imagery. It used a classical computer-vision pipeline and Hough circles to find approximately round rims. The later work expanded that base in several directions:

1. **Mars imagery and a planet selector.** The dashboard can now load a second, mixed-source Mars dataset. The Moon processing path stays the original Hough-circle path.
2. **A separate Mars detector.** Mars has a v1 Hough option for comparison and a v2 ellipse option, which is the default for Mars. The two planets do not have to use the same geometric assumption.
3. **Per-tile scale handling.** Moon tiles use a nominal 0.25 m/px. Mars tiles can store their own `m_per_px` in `tiles_config.json`; a scale-bar helper computes and saves it. If a Mars tile is not calibrated, the interface labels metre values “scale assumed.”
4. **More interpretable outputs.** The annotated image, hover details, pipeline stages, crater catalog, size/degradation plots, depth/hazard maps, and Moon-versus-Mars comparison expose what the pipeline produced.
5. **A crater-based Mission Planner.** Detected craters are expanded into a cost map. A distance transform selects a clear touchdown candidate, and A* can find a route from that point to a selected destination.
6. **Validation and documentation.** Offline evaluation compares detections with the exported YOLO boxes. A report records the current measured results and the limitations. The former dashboard Validation and Crater Inspector tabs were removed to keep the main interface more focused; the offline validation results remain useful for judging detector quality.

The central design choice throughout is explainability: each result can be traced to a visible image, a stated rule, or a stated assumption.

## 4. The image-processing path

The dashboard includes a Pipeline Stages view. Its images are useful because each operation solves a different problem and each can also introduce errors.

### 4.1 Start with the image and reduce speckle

The source tile is converted to grayscale. For the common processing branch, a 5 by 5 median filter replaces each pixel with the median of its neighborhood. This suppresses isolated bright or dark speckles while tending to keep a rim edge sharper than a comparable averaging blur.

Why this matters: crater rims are often thin curved boundaries, while orbital images can contain surface texture, compression, and small bright/dark details. A detector that responds to every texture mark will produce many false candidates. Denoising makes later edge decisions more manageable, but too much smoothing would erase small craters.

### 4.2 Make local contrast easier to inspect

The common branch applies CLAHE: Contrast-Limited Adaptive Histogram Equalization. It divides the image into an 8 by 8 grid of local regions and enhances contrast within those regions, with a clip limit of 2.0 by default. Unlike one global brightness stretch, this can reveal a rim in a locally dark or bright area without forcing the entire tile to share one contrast setting.

CLAHE is an aid to edge and shadow analysis, not a way to recover detail that is absent from the image. It can also amplify texture. The Mars v2 detector deliberately uses its own mild Gaussian blur and does not use CLAHE for its candidate-generation branch; CLAHE intermediates are still computed for the shared dashboard descriptors and display.

### 4.3 Find image boundaries with Canny

Canny edge detection looks for strong changes in brightness, which often occur at a crater rim. In the common branch, thresholds are set adaptively from the median enhanced intensity: approximately 0.55 times the median for the lower threshold and 1.45 times for the upper threshold.

The resulting image is an edge map, not a crater map. It also contains edges from streaks, small pits, shadows, and unrelated terrain. Later stages must decide which edge arrangements look crater-like.

### 4.4 Make a separate shadow image with Otsu thresholding

Otsu thresholding automatically chooses a grayscale cut that separates darker and brighter pixels. The project uses the inverse threshold so dark regions become foreground, then applies a small 3 by 3 elliptical morphological opening and closing. Opening removes some isolated specks; closing fills some small gaps.

This gives the dashboard a view of candidate shadowed interiors. A dark region is not automatically a crater, and a shadow mask does not reveal a true elevation profile. Its value is that users can compare the dark basin evidence with the rim edges and fitted geometry.

### 4.5 Detect lunar craters with Hough circles

The Moon detector runs OpenCV's Hough Gradient Circle Transform over a lightly blurred enhanced image. Hough transforms let image edges vote for possible circle centers and radii. They are useful when the target shape is approximately circular, even when the rim edge is incomplete.

The main Hough controls are:

- **Minimum and maximum radius (px):** candidate size bounds in the source image. Candidates outside them are not returned.
- **Detection sensitivity (`param2`):** the Hough accumulator threshold. A lower threshold usually admits more candidates and may increase false positives; a higher threshold is stricter and may miss weak rims.
- **Minimum distance (px):** discourages Hough from returning multiple centers close together.
- **Maximum detections:** caps the retained candidates after scoring.
- **`param1`:** the high Canny threshold used internally by Hough; the code uses the planet default.

Each Hough circle receives a rim-edge score from the edge pixels in a narrow band around its circumference. Candidates are considered from higher score to lower score, and near-duplicate centers are suppressed. This helps remove repeated fits of one rim, though two genuinely neighboring craters can still be confused.

The lunar defaults are min radius 6 px, max radius 85 px, `param2` 14, minimum distance 10 px, and a maximum of 80 detections. They are starting settings for this dataset, not universal lunar constants.

### 4.6 Detect Mars craters with ellipses

The Mars selector offers **v1 Hough** and **v2 Ellipse**. v2 is the default. Hough circles encode a center and one radius, so they fit round shapes naturally. In Mars tiles, a crater rim may look elongated because of image projection, viewing geometry, terrain, or the source image itself. A circle can then cut across the rim or create an oversized box. An ellipse has a major axis, a minor axis, and an orientation, so its bounding box can follow a visibly elongated outline more closely.

The current v2 process is:

1. Convert the tile to grayscale and apply mild Gaussian blur. The blur amount is scaled with image width. This reduces tiny texture while keeping the larger rim structure.
2. Run Canny on this Mars-specific image branch.
3. Apply morphological closing with an elliptical kernel scaled to the image dimensions. Closing can join gaps in a broken rim so it becomes a candidate region.
4. Find contours in the closed edge image. Contours long and large enough for fitting are passed to OpenCV's `fitEllipse`.
5. Optionally add candidates from Hough circles run at three smaller image-pyramid sizes. These candidates supplement the contour candidates; Hough is not the final shape assumption.
6. Measure support around each fitted ellipse. The rim is divided into 36 angular sectors. Coverage is the fraction of sectors containing edge pixels near the fitted rim.
7. Reject ellipses with a minor-to-major axis ratio below 0.4 or coverage below the configured threshold. The default minimum coverage is 0.6.
8. Merge duplicate boxes when their IoU is greater than 0.5, preferring the higher-scoring fit. Small nested craters can be kept when their radius is no more than 40% of the larger crater's radius.
9. If the filters reject every fit, show the highest-scoring candidates instead of silently presenting an empty result.

The Mars defaults include minimum radius 12 px, maximum radius 85 px, maximum 45 detections, blur scale 1.0, arc coverage 0.6, Canny ratio 0.6, and Hough supplement `param2` 28. For v2, `param2` mainly affects the optional Hough supplement; it does not replace the ellipse coverage filter. As in any candidate detector, these settings trade missed craters against false positives.

The ellipse method is related in broad outline to published candidate-region and ellipse-fitting approaches, including work by Bandeira and colleagues and Kim and Muller. This code is an empirical project implementation, not a reproduction of those papers. The project also retains Hough as a comparison method for Mars.

## 5. From pixels to measurements

After detection, the pipeline estimates crater diameter from image radius and scale:

> diameter (m) = 2 × radius (px) × scale (m/px)

For Mars ellipses, the reported radius is based on the geometric mean of the ellipse's semi-axes. This gives one representative size for the shared catalog and size classes, while the ellipse axes and bounding box preserve the elongated geometry for display and box evaluation.

The shared diameter classes are **Small** below 10 m, **Medium** from 10 m to below 30 m, and **Large** at 30 m or above. Because those are metre thresholds, classes depend directly on scale. An assumed Mars scale can make the class labels wrong even when the pixel geometry is reasonable.

### 5.1 Shadow-based depth and d/D

For each crater, the current depth routine examines a horizontal row through the detected center. It counts pixels below 70% of that row's mean intensity as shadow pixels. The fraction of the diameter occupied by those pixels is used as an approximate shadow length:

> estimated shadow length (px) = 2 × radius (px) × shadow fraction
>
> shadow length (m) = shadow length (px) × scale (m/px)
>
> estimated depth (m) = shadow length (m) × tan(sun elevation)
>
> d/D = estimated depth (m) ÷ diameter (m)

The sun-elevation control defaults to 30° for Moon and 45° for Mars, unless the selected tile has its own override. It changes the trigonometric conversion: higher assumed elevation gives a larger depth for the same estimated shadow length. This is a simple image heuristic. It does not measure the shadow direction or a full topographic profile, so the result should be read as a rough descriptor, not surveyed crater depth.

Mars depth and d/D are flagged low confidence by default because many Mars tiles are shaded or composited imagery without dependable cast-shadow geometry. Moon values can also be wrong when the row samples unrelated terrain or the illumination assumptions do not fit. The optional Depth Sanity Filter flags d/D values below 0.05 or above 0.25 as suspect. Its checkbox can exclude low-confidence values from the mean-depth KPI; it does not delete those craters from the catalog.

### 5.2 Freshness and degradation

The freshness index combines two image measurements:

> freshness = 0.6 × rim sharpness + 0.4 × floor smoothness

Rim sharpness is the fraction of edge pixels in a narrow band around the detected circle. Floor smoothness is one minus the normalized standard deviation of enhanced-image intensity inside a disk covering 60% of the radius. The index ranges roughly from 0 to 1. Larger values support a sharper rim and smoother measured floor under this rule.

The project maps that score to four labels: Fresh above 0.48; Slightly Degraded above 0.32; Moderately Degraded above 0.18; otherwise Heavily Degraded. These labels describe the detector's image-based heuristic. They are not crater ages, formal geologic units, or a validated chronology. Lighting, blur, resolution, and image source can affect them.

### 5.3 Detection IDs, annotations, and catalog

IDs link the numbered marks on the annotated tile to rows in the crater catalog. Hovering over a detection shows its brief measurements and classification. Moon detections use purple; Mars detections use bright yellow with a thin black outline for contrast. Dataset ground-truth boxes, when enabled, use cyan so the reference label can be distinguished from the detector output.

The annotation style can show bounding boxes, circles/ellipse outlines, or both. The catalog includes image coordinates, pixel radius, diameter, size class, estimated depth, d/D, degradation, freshness, rim coverage where available, and depth-confidence flag. It can be exported as CSV for later analysis. The batch view applies the active settings over a selected planet's tiles; the Moon-vs-Mars view shows their output distributions side by side.

## 6. What each dashboard adjustment changes

| Control | What it changes | How to interpret it |
|---|---|---|
| Planet | Chooses Moon or Mars image collection and defaults | Moon stays on the original Hough path; Mars can use Hough or ellipse v2 |
| Tile | Chooses the source image and matching annotation/config | Tile labels and all image-coordinate measurements refer to this selection |
| Mars detector | Selects v1 Hough or v2 Ellipse | v2 is the default; v1 is useful as a comparison, not an automatic truth |
| Min/max crater radius | Limits detector candidates in pixels | Set these to the image-space sizes of interest; too narrow excludes real craters |
| Detection sensitivity (`param2`) | Changes Hough candidate threshold | Lower tends to return more, including false positives; higher tends to be stricter |
| Minimum distance | Changes separation between Hough circle centers | Larger spacing reduces clusters but can suppress close real craters |
| Maximum detections | Caps retained candidates | A low cap can omit weak or smaller detections even if proposed |
| CLAHE clip limit | Changes common-branch local contrast enhancement | More contrast may expose rims and shadows, but can amplify texture |
| Spatial resolution (m/px) | Converts pixels to metres and areas/densities | A stored tile scale takes precedence; otherwise the shown fallback is used |
| Calibrate from scale bar | Converts bar pixels and bar kilometres into m/px and saves it to the Mars tile config | Formula: m/px = bar kilometres × 1000 ÷ bar pixels. A wrong bar measurement shifts every metre result |
| Sun elevation | Changes the shadow-depth calculation | It affects depth and d/D, not the detected rim itself |
| Annotation style / IDs / GT | Changes display and label overlays | GT is the dataset's YOLO boxes, not a second detector output |
| Suppress dune/streak false positives | Optionally removes detections below a rim-edge-coverage threshold | Helpful against streak-like shapes, but may also remove broken real rims |
| Depth Sanity Filter | Optionally excludes suspect d/D values from mean depth | It only filters the mean KPI; catalog entries remain available |

If you adjust detector controls, the counts and derived plots can change. If you adjust scale, pixel detections do not move, but their metre values, size classes, physical area, density, hazard radii, and routes can change.

## 7. The Mission Planner, from map to route

The Mission Planner asks a simple question: given this tile and the crater hazards currently known to the app, where is a clear landing cell, and what route could a rover take from there?

### 7.1 Build the map

The image is rasterized into a grid with a cell size of at least about 1 m; when the source image has a coarser ground sample than 1 m/px, the map cannot create finer real detail than that source scale. Each detected crater is projected into metre coordinates. Ellipses use their major semi-axis for the planner's conservative circular hazard footprint, so an elongated crater is not treated as safe along its long edge. Every detection counts as a hazard, even when confidence information is unavailable. The user can also add a missed crater by entering its image-pixel center and radius; these manual hazards last for the current app session.

Each crater creates three map regions:

- **Interior / blocked:** infinite traversal cost. The rover cannot cross it.
- **High-cost rim zone:** the base safety buffer plus 0.25 times the crater radius, expanded for the rover footprint. The base buffer defaults to 1 m. The cost increases with the freshness index: `50 × (1 + freshness)`.
- **Medium-cost outer halo:** extends a further distance set by the Outer caution zone slider, default 5 m. Its cost is `5 × (1 + 0.5 × freshness)`.

Where buffers overlap, the more hazardous cost wins. A fresh-looking crater is assigned higher costs because sharp rims and surrounding ejecta can represent a more concerning hazard in this simplified model. It is still a heuristic, not a measured rock field. A roughly 2 m strip at the image boundary is also reserved so the route does not exploit the edge as a shortcut.

The cost overlay colors the blocked crater/rover footprint red, the high-cost rim region orange, and the medium-cost halo yellow. The green marker is the touchdown candidate. “Hazard layers only” shows the same cost regions on a dark background.

### 7.2 Choose a touchdown point

The distance transform computes, for each grid cell, its distance to the nearest crater keep-out region. The recommended landing point is the available cell with the greatest clearance. The green marker and footprint circle show its location and the configured lander radius.

The default **lander footprint radius** is 2 m; it is the radius of the circular footprint used to describe the lander. The **landing safety margin** defaults to 1 m and adds extra clearance around the vehicle. Rover dimensions are also used because the chosen site becomes the rover's start: default rover width is 1.5 m, length is 2 m, and clearance margin is 0.5 m. The code approximates its circular radius as half the diagonal of the width/length rectangle plus that margin. The required clearance is the larger of the lander radius and rover radius, plus the landing margin.

“Recommend the safest available site” selects the maximum-clearance point. “Choose a touchdown coordinate” lets the user request an image coordinate, then snaps it to the nearest hazard-free grid cell. The app reports whether the selected point meets the requested clearance. If there is no fully hazard-free region, it shows the best available candidate with a warning rather than pretending it meets the requirement.

Touchdown clearance is measured from the crater keep-out boundaries represented by this map. It does not account for slope, rocks, undetected craters, surface strength, or navigation uncertainty.

### 7.3 Select a rover destination

The rover starts at the touchdown point. The destination can be:

- **Land only:** no rover route is requested.
- **Visit the largest crater rim:** selects the largest detected crater and ends at a traversable cell just outside its rim buffer.
- **Visit the nearest crater by size:** choose Small, Medium, or Large; the planner selects the closest detected crater in that class and targets a point outside its buffer.
- **Reach image coordinates:** enter x and y in pixels; the requested point is snapped to the nearest hazard-free cell.

The objective makes “goal” explicit. The crater center is not a valid route endpoint because it lies inside the blocked area; a crater-rim objective means reach the safe side of the keep-out boundary.

### 7.4 A* and the route policies

The router is an eight-connected A* search. It can move horizontally/vertically at a distance of one cell, or diagonally at √2 cells. It averages the cost of the current and next cells for each step, and it prevents diagonal moves from slipping between two blocked cells. The A* heuristic is straight-line grid distance multiplied by the lowest available cell cost.

All policies keep crater interiors blocked. They differ in how much they avoid the caution zones:

| Policy | Rim/halo cost multiplier | Turn penalty multiplier | Practical effect |
|---|---:|---:|---|
| Safest | 2.0 | 1.0 | Prefers a wider berth around costly zones and stronger smoothing |
| Balanced | 1.0 | 0.45 | Compromise between distance, hazard cost, and sharp turns |
| Shortest | 0.15 | 0.1 | More willing to cross costly rim/halo cells; interiors remain blocked |

The **rover width**, **rover length**, and **rover clearance margin** expand crater zones to account for vehicle size. The **turn radius** defaults to 3 m and adds an approximate penalty for direction changes. It does not simulate steering, wheel slip, acceleration, or a real vehicle's kinematics.

The selected route is drawn as a bright line with a dark under-stroke, the direct start-to-destination segment is dashed red, the touchdown is green with a footprint circle, and the destination is an orange diamond. Crater detections keep their planet color. The straight line is a comparison, not a proposed safe route.

### 7.5 Read the route metrics

- **Planned path length:** sum of A* steps in metres, including diagonal steps.
- **Straight-line length:** direct Euclidean distance between route start and endpoint. It is the shortest geometric comparison, even if it crosses hazards.
- **Detour:** `(planned length / straight-line length − 1) × 100%`. A positive value shows the extra distance taken to route around costs and obstacles.
- **Known craters crossed, straight / route:** number of crater interiors intersected by each line/path using the known crater geometry and rover radius. “Known” matters: an undetected crater is absent from the calculation.
- **Minimum rim clearance:** the smallest route distance outside any known crater radius plus its safety buffer and rover radius. It summarizes how closely the route approaches the modeled rim keep-outs.
- **Touchdown clearance:** the distance-transform clearance at the selected landing cell, compared with the required vehicle clearance.
- **Map scale:** the active m/px value and whether it is calibrated, a dataset default, or assumed.

Every metre-valued Mars result marked “scale assumed” inherits the fallback scale until that tile is calibrated. In that case route lengths, footprint sizes, clearances, and hazard widths are illustrative.

## 8. What research already does, and what this project adds

Planetary crater research includes several different scales and goals. Robbins and Hynek's Mars inventory and Robbins's lunar database catalog craters at global scale for population studies. Bandeira and colleagues studied shape and texture features; Kim and Muller assessed automated Mars crater detection; Galloway and colleagues applied Hough transforms to crater detection and counting. More recent lunar work by Sinha and colleagues uses deep convolutional neural networks on Chandrayaan-2 TMC-2 imagery. Crater depth studies use better constrained topography, photoclinometry, or geomorphic observations than a single image scanline. Operational landing systems use much richer terrain-relative navigation and hazard information; NASA/JPL's Jezero hazard-map material is one example of that operational context.

This project uses those ideas as context and engineering ingredients, but its contribution is an integrated, inspectable demonstration rather than a new detection algorithm. It puts conventional image processing, two planet-specific geometric detectors, approximate physical descriptors, a crater hazard map, and a route demonstration into one interactive flow. Users can see why the software produced an outline and what the next calculation did with it. The project also records real mission context: ISRO documented a crater encountered by Pragyan during its traverse, and NASA/JPL illustrates how hazards and safer terrain matter in the Jezero landing area. Those examples explain why local surface interpretation is useful; they do not mean this prototype performs those missions' navigation tasks.

The trade-off is that the project does not have the scale, training/evaluation depth, global coverage, or terrain data of those research and mission systems. It is useful for learning, prototyping, and generating candidate outputs that another system or a reviewer can inspect. The project should not claim it matches the accuracy of global catalogs or mission navigation systems.

## 9. What the current Moon-versus-Mars results say

The current experiment summary used 20 lunar tiles and 60 Mars tiles. Moon tiles use a nominal 0.25 m/px. Mars tiles are mixed-source, and only 1 of 60 has a calibrated scale in the configuration; the other 59 use a 5 m/px fallback. The difference in source, resolution, and calibration means the counts are not a controlled comparison of the Moon and Mars as planets.

The current catalogs contain 626 Moon Hough detections and 1,145 Mars v2 ellipse detections. In this run the Moon detections divide into 399 Small, 217 Medium, and 10 Large; the Mars detections classify as 1,145 Large and zero in the other classes because the assumed 5 m/px scale turns even the minimum detected Mars radii into large diameters under the shared 10 m/30 m cutoffs. This is a warning about scale and class interpretation, not evidence that Mars has no small craters.

The held-out box evaluation is also modest. Greedy one-to-one matching at IoU 0.5 gave:

| Detector and held-out set | Precision | Recall | F1 | Mean IoU of matched boxes |
|---|---:|---:|---:|---:|
| Moon v1 Hough | 0.179 | 0.086 | 0.116 | 0.633 |
| Mars v1 Hough | 0.062 | 0.075 | 0.068 | 0.566 |
| Mars v2 Ellipse | 0.070 | 0.093 | 0.080 | 0.632 |

The ellipse version is better than Mars Hough on these held-out labels by F1 and mean matched-box IoU, but both Mars scores are low. The detector still misses most labeled craters and produces many predictions that do not match a label. The Moon score is also low. The box-matching experiment is a useful baseline and a way to compare variants; it does not certify mission readiness.

What can we conclude? In these tiles and with these settings, ellipses model some Mars crater boxes better than circles, while the whole detection problem remains difficult. We cannot conclude that Mars has more craters than the Moon, that one planet has older terrain, or that a particular detector is generally superior across imagery. The tile sample, scales, image sources, and detection recall do not support those claims.

## 10. How the outputs could help real systems

The useful handoff is not just an annotated screenshot. It is a set of data products:

1. **Candidate crater records:** IDs, image coordinates, fitted circle/ellipse geometry, and bounding boxes. These can be reviewed by a person or used to seed a larger catalog.
2. **Image-derived attributes:** diameter estimate, d/D estimate, rim coverage, freshness score, and degradation class, each with its assumptions. These could be attached to map features or used to prioritize manual review.
3. **Hazard raster:** blocked interiors, high-cost rim buffers, and medium-cost halos. A navigation or GIS system could combine these with slope, rocks, elevation, and uncertainty layers.
4. **Candidate route and metrics:** an example touchdown, route polyline, direct-line comparison, and clearance/detour summary. These can be compared with richer terrain maps or used to explain why a route planner selected a path.
5. **CSV and image overlays:** shareable analysis results for a report, further scripts, or human quality control.

To make these products operationally useful, a downstream system would need georeferenced coordinates, calibrated image geometry, reliable scale, validated detections, uncertainty estimates, complete hazard layers, and vehicle-specific constraints. The current output is a prototype input layer for that larger process, not the final decision.

## 11. Conclusion: what the project demonstrates

The project demonstrates how a classical vision pipeline can connect image evidence to an interpretable planning example. Median filtering reduces small noise; CLAHE makes local contrast easier to work with; Canny highlights boundaries; Otsu creates a dark-region view; Hough proposes lunar circles; and ellipse fitting provides Mars candidates that can follow elongated rims. The catalog converts those candidates into approximate physical and morphological descriptors. The Mission Planner then turns the known detections into a cost grid, selects a high-clearance touchdown candidate, and runs A* to a clear destination.

The results are most useful as a transparent baseline. They show where assumptions enter, how different detector geometry affects the output, and why validation, calibrated scale, and richer terrain data matter. Before use by another scientific or navigation application, detection quality would need improvement and validation on well-described imagery; depth would need topographic verification; and the planner would need slope, rocks, terrain uncertainty, and vehicle dynamics. That is the natural next step from an educational end-to-end demo toward a dependable planetary mapping component.

## 12. Short glossary

- **Pixel (px):** one image sample; not a physical distance until scale is known.
- **m/px:** metres represented by one image pixel.
- **Ground truth (GT):** the dataset's manually supplied bounding boxes used for reference/evaluation.
- **Canny edge:** a pixel marked as a likely boundary due to brightness change.
- **Morphology:** neighborhood operations such as opening/closing that remove small artifacts or bridge small gaps.
- **Hough transform:** a voting method that proposes geometric shapes such as circles from edge pixels.
- **Ellipse fit:** a geometric approximation described by center, two axes, and orientation.
- **IoU:** intersection over union; overlap area divided by combined area for two boxes. At 0.5, a predicted box must overlap a label by at least half under this metric to match.
- **Precision:** fraction of predicted detections that match labels.
- **Recall:** fraction of labeled craters that the detector finds.
- **F1:** harmonic balance of precision and recall.
- **d/D:** estimated depth divided by diameter; a scale-free ratio, but only as reliable as the depth estimate.
- **A*:** a shortest-path search that combines distance traveled so far with an estimate of remaining distance, here also accounting for cell costs.
- **Cost map:** grid assigning each cell blocked or a relative travel penalty; it is not a probability map or measured terrain slope map.
