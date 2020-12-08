# Changelog

## 2.0.1.post1 release (2019-01-09)
Framework-agnostic mAP computation python API on bounding boxes and polygons for fast and easy validation of segmentation and detections models.

> **Warning**: The 2.x family introduced major fixes to metrics computed as well as significant paradigm changes as to what should be penalized and how. As a results, metrics computed in the `xview` mode are no longer identical to the metrics computed with the xview scoring codebase.
> As a consequence, the 1.x family is deprecated and metrics computed with the 1.x family should not be trusted anymore.

### Features
* Change readme and CI stack

### Known bugs
* F-beta helper is incompatible with alternative match method (#21)

---

## 2.0.1.post0 release (2019-01-02)
Framework-agnostic mAP computation python API on bounding boxes and polygons for fast and easy validation of segmentation and detections models.

> **Warning**: The 2.x family introduced major fixes to metrics computed as well as significant paradigm changes as to what should be penalized and how. As a results, metrics computed in the `xview` mode are no longer identical to the metrics computed with the xview scoring codebase.
> As a consequence, the 1.x family is deprecated and metrics computed with the 1.x family should not be trusted anymore.

### Features
* Add Changelog in documentation

### Known bugs
* F-beta helper is incompatible with alternative match method (#21)

---

## 2.0.1 release (2018-12-05)
Framework-agnostic mAP computation python API on bounding boxes and polygons for fast and easy validation of segmentation and detections models.

> **Warning**: The 2.x family introduced major fixes to metrics computed as well as significant paradigm changes as to what should be penalized and how. As a results, metrics computed in the `xview` mode are no longer identical to the metrics computed with the xview scoring codebase.
> As a consequence, the 1.x family is deprecated and metrics computed with the 1.x family should not be trusted anymore.


### Features
* Speed improvements (#19)

### Fixes
* Fix get_type_and_convert for str ndarray and possibly other hashable labels (#24)
* Replace corrupted MatchEngine xView implementation equality test with scalability test (#19)


### Known bugs
* F-beta helper is incompatible with alternative match method (#21)

---

## 2.0.0 release (2018-11-29)
Framework-agnostic mAP computation python API on bounding boxes and polygons for fast and easy validation of segmentation and detections models.

> **Warning**: The 2.x family introduced major fixes to metrics computed as well as significant paradigm changes as to what should be penalized and how. As a results, metrics computed in the `xview` mode are no longer identical to the metrics computed with the xview scoring codebase.
> As a consequence, the 1.x family is deprecated and metrics computed with the 1.x family should not be trusted anymore.


### Breaking
* Add default values for threshold and match_algorithm in MeanAveragePrecisionMetric. Add FutureWarning if providing
  both those and a custom match engine to warn that in the future, this will raise a ValueError.
* Split `MatchEngine` into an abstract base class `MatchEngineBase` and an IoU based one `MatchEngineIoU`
  corresponding to former `MatchEngine`
* `MatchEngine` and `MeanAveragePrecision` internals uses new Geometry objects
* `add_confidence_from_max_IoU` uses new `Geometry` objects


### Features
* Add Mean F-beta class metric helper
* Added new property to MeanAveragePrecisionMetric: ground_truth_labels (#20)
* Documentation improvements
* Add non-unitary match method and update mAP computer precision and recall computing method to correctly handle non-unitary match case. (#10)
* Add attributes (#11):
  * ``self.number_true_detection_per_class`` -> A dict of the number of detection matched to a ground truth
  * ``self.number_false_detection_per_class`` -> A dict of the number of detection not matched to a ground truth
  * ``self.number_found_ground_truth_per_class`` -> A dict of the number of ground truth matched to a detection
  * ``self.number_missed_ground_truth_per_class`` -> A dict of the number of ground truth not matched to a detection
* Add autocorrect_invalid_geometry kwarg to handle systemic self crossing edge case (#9)
* Add geometry classes as wrappers around shapely for internal use (#4)
* Add metrics computation with points (#3)
* Add python 2.7 testing


### Fixes
* Fixed Bug/Feature with MatchEnginePointInBox (#13)
* Fixed ValueError: all the input arrays must have same number of dimensions (#23)
* Changed raise NotImplementedError to return NotImplemented on geometric object binary operation
* Fixed metaclass problem with py3 (#18)
* Fixes empty array input bug (#5). Add mAP computation consistency on empty tiles
* Explicitely cast label array to an np.int32 array and explicitely cast out dict
  keys to int to ensure json serialisation
* Fix empty input case to avoid over decrease (#7)
* Fix a miscalculation of the number of ground truths when no detections
  where present for a given class.


### Known bugs
* Fix get_type_and_convert for str ndarray and possibly other hashable labels (#24)
* F-beta helper is incompatible with alternative match method (#21)

---

## 2.0.0a3 release (2018-10-09)
Framework-agnostic mAP computation python API on bounding boxes and polygons for fast and easy validation of segmentation and detections models.

> **Warning**: The 2.x family introduced major fixes to metrics computed as well as significant paradigm changes as to what should be penalized and how. As a results, metrics computed in the `xview` mode are no longer identical to the metrics computed with the xview scoring codebase.
> As a consequence, the 1.x family is deprecated and metrics computed with the 1.x family should not be trusted anymore.

### Features
* Add non-unitary match method and update mAP computer precision and recall computing method to correctly handle non-unitary match case. (#10)
* Add attributes (#11):
  * ``self.number_true_detection_per_class`` -> A dict of the number of detection matched to a ground truth
  * ``self.number_false_detection_per_class`` -> A dict of the number of detection not matched to a ground truth
  * ``self.number_found_ground_truth_per_class`` -> A dict of the number of ground truth matched to a detection
  * ``self.number_missed_ground_truth_per_class`` -> A dict of the number of ground truth not matched to a detection

---

## 2.0.0a2 release (2018-09-24)
Framework-agnostic mAP computation python API on bounding boxes and polygons for fast and easy validation of segmentation and detections models.

> **Warning**: The 2.x family introduced major fixes to metrics computed as well as significant paradigm changes as to what should be penalized and how. As a results, metrics computed in the `xview` mode are no longer identical to the metrics computed with the xview scoring codebase.
> As a consequence, the 1.x family is deprecated and metrics computed with the 1.x family should not be trusted anymore.

### Features
* Add autocorrect_invalid_geometry kwarg to handle systemic self crossing edge case (#9)

---

## 2.0.0a1 release (2018-09-21)
Framework-agnostic mAP computation python API on bounding boxes and polygons for fast and easy validation of segmentation and detections models.

> **Warning**: The 2.x family introduced major fixes to metrics computed as well as significant paradigm changes as to what should be penalized and how. As a results, metrics computed in the `xview` mode are no longer identical to the metrics computed with the xview scoring codebase.
> As a consequence, the 1.x family is deprecated and metrics computed with the 1.x family should not be trusted anymore.

### Breaking
* Split `MatchEngine` into an abstract base class `MatchEngineBase` and an IoU based one `MatchEngineIoU` corresponding to former `MatchEngine`
* `MatchEngine` and `MeanAveragePrecision` internals uses new Geometry objects
* `add_confidence_from_max_IoU` uses new `Geometry` objects

### Features
* Add geometry classes as wrappers around shapely for internal use (#4)
* Add metrics computation with points (#3)
* Add python 2.7 testing

### Fixes
* Fixes empty array input bug as #5. Add mAP computation consistency on empty tiles
* Explicitely cast label array to an np.int32 array and explicitely cast out dict
  keys to int to ensure json serialisation
* Fix empty input case to avoid over decrease as stated in #7
* Fix a miscalculation of the number of ground truths when no detections
  where present for a given class.

---

## 1.0.0 release (2018-07-25)

Framework-agnostic mAP computation python API for fast and easy validation.

### Features
* Compute mAP from bounding boxes or polygons
* Compute AP per class
* Compute Precision per class
* Compute Recall per class
* Use xView or Coco matching algorithms
