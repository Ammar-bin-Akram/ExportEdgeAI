"""
OpenCV-based cosmetic defect detection for mango export quality assessment.

Designed for **preprocessed ROI images** (224×224) that have already been
through ImageProcessor (denoise → colour-correct → sharpen → CLAHE → resize).

Detects:
    1. Dark / black spots  (anthracnose, bacterial spots)
    2. Brown spots / patches  (bruises, sap burn, blemishes)
    3. Colour uniformity

All thresholds auto-scale with image resolution so the detector also works
on raw high-resolution images if needed.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Reference resolution that the base thresholds are tuned for.
_REF_AREA = 224 * 224  # 50 176 px


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DefectRegion:
    """A single detected defect on the mango surface."""
    type: str                                   # 'dark_spot' | 'brown_spot'
    contour: np.ndarray
    area: float                                 # in pixels at input resolution
    area_pct: float                             # % of mango surface
    center: Tuple[int, int]
    severity: str                               # 'minor' | 'moderate' | 'severe'
    bounding_box: Tuple[int, int, int, int]     # x, y, w, h
    confidence: float
    mean_intensity: float = 0.0


@dataclass
class DefectAnalysis:
    """Complete defect analysis results."""
    total_defect_area: float
    total_defect_percentage: float          # % of mango surface
    mango_area: float                       # mango surface area (px)
    defect_count: int
    dark_spot_count: int
    brown_spot_count: int
    defect_regions: List[DefectRegion]
    color_uniformity_score: float           # 0–100, higher = more uniform
    surface_quality_score: float            # 0–100, higher = better
    export_grade_impact: str                # 'minimal' | 'moderate' | 'significant'
    processing_time: float


# ── Detector ──────────────────────────────────────────────────────────────────

class MangoDefectDetector:
    """
    Cosmetic defect detection for mango ROIs.

    Works on **preprocessed 224×224 images** (primary use-case) as well as
    raw high-resolution images.  Thresholds are defined at 224×224 scale and
    automatically multiplied by ``image_area / 224² `` for larger inputs.

    The detector expects the mango to fill *most* of the ROI (as output by
    ``extract_roi`` or ``ROIExtractor``).  A lightweight mask is used to
    exclude residual background / conveyor-belt edges.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = {
            # ── Background mask (lightweight, ROI-oriented) ──
            'bg_sat_min': 20,               # min saturation to be mango
            'bg_hue_low': 10,               # mango hue lower bound
            'bg_hue_high': 100,             # mango hue upper bound
            'bg_morph_kernel': 5,           # morph kernel (scales with image)
            'bg_erode_kernel': 3,           # edge erosion (scales with image)
            'bg_min_area_ratio': 0.10,      # mask must cover ≥10 % of image

            # ── Dark / black spot detection ──
            # Thresholds below are at 224×224 scale; auto-scaled for larger.
            'dark_threshold': 55,           # CLAHE-grey intensity cut-off
            'dark_min_area': 12,            # min spot area (px @224)
            'dark_max_area': 6000,          # max spot area (px @224)
            'dark_max_hsv_v': 120,          # reject if mean V above this
            'dark_noise_kernel': 3,         # opening kernel
            'dark_close_kernel': 3,         # closing kernel

            # ── Brown spot detection (HSV) ──
            'brown_hue_low': 4,
            'brown_hue_high': 20,
            'brown_sat_low': 60,
            'brown_sat_high': 210,
            'brown_val_low': 25,
            'brown_val_high': 130,
            'brown_min_area': 18,           # px @224
            'brown_noise_kernel': 3,
            'brown_close_kernel': 3,

            # ── Colour uniformity ──
            # (uses coefficient-of-variation, no config threshold needed)

            # ── Severity thresholds (px @224) ──
            'sev_moderate_area': 60,
            'sev_severe_area': 200,
            'sev_severe_intensity': 35,
            'sev_moderate_intensity': 55,

            # ── Export grading ──
            'grade_a_max_pct': 2.0,
            'grade_b_max_pct': 5.0,
            'severe_area_threshold': 120,   # px @224

            # ── Preprocessing control ──
            'skip_clahe': True,             # True when input already has CLAHE
        }

        if config:
            self.config.update(config)

    # ──────────────────────────────────────────────────────────────────────
    #  Scale helper
    # ──────────────────────────────────────────────────────────────────────
    def _scale(self, base_value: float, image_area: int) -> float:
        """Scale a pixel-area threshold from 224×224 to the actual image."""
        return base_value * (image_area / _REF_AREA)

    def _kernel(self, base_size: int, image_area: int) -> int:
        """Scale a kernel size; always odd ≥ 1."""
        s = max(1, int(round(base_size * (image_area / _REF_AREA) ** 0.5)))
        return s if s % 2 == 1 else s + 1

    # ──────────────────────────────────────────────────────────────────────
    #  Background mask (lightweight)
    # ──────────────────────────────────────────────────────────────────────
    def create_mango_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Create a binary mask isolating the mango from residual background.

        For preprocessed ROIs the mango fills most of the frame, so this
        is intentionally lightweight: HSV saturation + hue range, small
        morphology, keep largest contour.

        Returns:
            Binary mask (255=mango, 0=background).
        """
        img_area = image.shape[0] * image.shape[1]

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, _ = cv2.split(hsv)

        # Saturation mask — foreground (mango) is more colourful than belt
        sat_mask = cv2.inRange(s, self.config['bg_sat_min'], 255)

        # Hue mask — green / yellow / orange tones only
        hue_mask = cv2.inRange(h, self.config['bg_hue_low'],
                               self.config['bg_hue_high'])

        mask = cv2.bitwise_and(sat_mask, hue_mask)

        # Morphological close + open
        mk = self._kernel(self.config['bg_morph_kernel'], img_area)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)

        # Keep only the largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > img_area * self.config['bg_min_area_ratio']:
                mask = np.zeros_like(mask)
                cv2.drawContours(mask, [largest], -1, 255, -1)

        # Light erosion to pull away from the boundary
        ek = self._kernel(self.config['bg_erode_kernel'], img_area)
        mask = cv2.erode(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek, ek)),
            iterations=1,
        )

        # Fallback: if the mask is too small, assume the whole image is mango
        if cv2.countNonZero(mask) < img_area * self.config['bg_min_area_ratio']:
            logger.debug("Mask too small — falling back to full-image mask")
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

        return mask

    # ──────────────────────────────────────────────────────────────────────
    #  Preprocessing
    # ──────────────────────────────────────────────────────────────────────
    def preprocess_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Convert to required colour spaces.

        If the image has already been through ImageProcessor (CLAHE etc.),
        set ``skip_clahe=True`` in config to avoid double enhancement.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab  = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        if self.config.get('skip_clahe', True):
            # Image already has CLAHE from ImageProcessor — just use blurred
            enhanced = blurred
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(blurred)

        return {
            'original': image,
            'gray': gray,
            'hsv': hsv,
            'lab': lab,
            'enhanced': enhanced,
        }

    # ──────────────────────────────────────────────────────────────────────
    #  Dark / black spot detection
    # ──────────────────────────────────────────────────────────────────────
    def detect_dark_spots(self, enhanced_gray: np.ndarray,
                          mango_mask: np.ndarray,
                          hsv_image: np.ndarray) -> List[DefectRegion]:
        """Detect genuinely dark / black spots on the mango surface."""
        dark_spots: List[DefectRegion] = []
        img_area = enhanced_gray.shape[0] * enhanced_gray.shape[1]
        mango_area = max(1.0, float(cv2.countNonZero(mango_mask)))

        min_area = self._scale(self.config['dark_min_area'], img_area)
        max_area = self._scale(self.config['dark_max_area'], img_area)

        # Threshold for dark pixels
        _, dark_bin = cv2.threshold(
            enhanced_gray, self.config['dark_threshold'], 255,
            cv2.THRESH_BINARY_INV,
        )
        dark_bin = cv2.bitwise_and(dark_bin, mango_mask)

        # Morphological clean-up (scaled kernels)
        nk = self._kernel(self.config['dark_noise_kernel'], img_area)
        ck = self._kernel(self.config['dark_close_kernel'], img_area)
        dark_bin = cv2.morphologyEx(
            dark_bin, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nk, nk)))
        dark_bin = cv2.morphologyEx(
            dark_bin, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck)))

        contours, _ = cv2.findContours(dark_bin, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area <= area <= max_area):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            center = (x + w // 2, y + h // 2)

            spot_mask = np.zeros_like(enhanced_gray)
            cv2.drawContours(spot_mask, [cnt], -1, 255, -1)
            mean_int = cv2.mean(enhanced_gray, mask=spot_mask)[0]

            # Reject green-but-dark regions using HSV colour validation
            mean_hsv = cv2.mean(hsv_image, mask=spot_mask)
            if mean_hsv[2] > self.config['dark_max_hsv_v'] and mean_hsv[1] > 50:
                continue  # coloured skin, not a true dark spot

            severity   = self._classify_severity(area, mean_int, img_area)
            confidence = self._spot_confidence(cnt, area)
            area_pct   = (area / mango_area) * 100.0

            dark_spots.append(DefectRegion(
                type='dark_spot', contour=cnt, area=area, area_pct=area_pct,
                center=center, severity=severity,
                bounding_box=(x, y, w, h),
                confidence=confidence, mean_intensity=mean_int,
            ))

        return dark_spots

    # ──────────────────────────────────────────────────────────────────────
    #  Brown spot detection
    # ──────────────────────────────────────────────────────────────────────
    def detect_brown_spots(self, hsv_image: np.ndarray,
                           mango_mask: np.ndarray) -> List[DefectRegion]:
        """Detect brown spots / patches using an HSV colour range."""
        brown_spots: List[DefectRegion] = []
        img_area = hsv_image.shape[0] * hsv_image.shape[1]
        mango_area = max(1.0, float(cv2.countNonZero(mango_mask)))
        min_area = self._scale(self.config['brown_min_area'], img_area)

        lower = np.array([self.config['brown_hue_low'],
                          self.config['brown_sat_low'],
                          self.config['brown_val_low']])
        upper = np.array([self.config['brown_hue_high'],
                          self.config['brown_sat_high'],
                          self.config['brown_val_high']])

        brown_bin = cv2.inRange(hsv_image, lower, upper)
        brown_bin = cv2.bitwise_and(brown_bin, mango_mask)

        nk = self._kernel(self.config['brown_noise_kernel'], img_area)
        ck = self._kernel(self.config['brown_close_kernel'], img_area)
        brown_bin = cv2.morphologyEx(
            brown_bin, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nk, nk)))
        brown_bin = cv2.morphologyEx(
            brown_bin, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck)))

        contours, _ = cv2.findContours(brown_bin, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        sev_mod  = self._scale(self.config['sev_moderate_area'], img_area)
        sev_sev  = self._scale(self.config['sev_severe_area'], img_area)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            center  = (x + w // 2, y + h // 2)
            sev     = ('minor' if area < sev_mod
                       else 'moderate' if area < sev_sev
                       else 'severe')
            conf     = self._spot_confidence(cnt, area)
            area_pct = (area / mango_area) * 100.0

            brown_spots.append(DefectRegion(
                type='brown_spot', contour=cnt, area=area, area_pct=area_pct,
                center=center, severity=sev,
                bounding_box=(x, y, w, h), confidence=conf,
            ))

        return brown_spots

    # ──────────────────────────────────────────────────────────────────────
    #  Colour uniformity
    # ──────────────────────────────────────────────────────────────────────
    def analyze_color_uniformity(self, image: np.ndarray,
                                 mango_mask: np.ndarray) -> float:
        """Colour uniformity of the mango surface in LAB space (0–100).

        Uses coefficient-of-variation style scoring so the result
        is comparable across raw and CLAHE-enhanced images.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        px = lab[mango_mask > 0]
        if len(px) == 0:
            return 0.0

        # Per-channel standard-deviation normalised by channel mean
        cvs = []
        for ch in range(3):
            vals = px[:, ch].astype(np.float64)
            mu = np.mean(vals)
            if mu < 1.0:
                continue
            cvs.append(np.std(vals) / mu)

        if not cvs:
            return 0.0

        avg_cv = float(np.mean(cvs))
        # Map coefficient-of-variation to a 0–100 score.
        # CV ~0.05 → perfectly uniform, CV ~0.40 → very patchy
        score = max(0.0, 100.0 * (1.0 - avg_cv / 0.40))
        return min(100.0, score)

    # ──────────────────────────────────────────────────────────────────────
    #  Scoring helpers
    # ──────────────────────────────────────────────────────────────────────
    def calculate_surface_quality_score(self, regions: List[DefectRegion]) -> float:
        """Surface quality 0–100 (higher = better)."""
        if not regions:
            return 100.0
        sev = sum(1 for d in regions if d.severity == 'severe')
        mod = sum(1 for d in regions if d.severity == 'moderate')
        mnr = sum(1 for d in regions if d.severity == 'minor')
        return max(0.0, 100.0 - sev * 15 - mod * 7 - mnr * 3)

    def assess_export_grade_impact(self, defect_pct: float,
                                   regions: List[DefectRegion],
                                   img_area: int) -> str:
        """'minimal' | 'moderate' | 'significant'."""
        severe = [d for d in regions if d.severity == 'severe']
        large  = [d for d in regions
                  if d.area > self._scale(self.config['severe_area_threshold'], img_area)]

        if (defect_pct > self.config['grade_b_max_pct']
                or len(severe) > 1 or len(large) > 0):
            return 'significant'
        if defect_pct > self.config['grade_a_max_pct'] or len(severe) > 0:
            return 'moderate'
        return 'minimal'

    # ──────────────────────────────────────────────────────────────────────
    #  Main entry point
    # ──────────────────────────────────────────────────────────────────────
    def detect_defects(self, image: np.ndarray, *,
                       mask: Optional[np.ndarray] = None,
                       save_debug: bool = False,
                       debug_path: Optional[Path] = None) -> DefectAnalysis:
        """
        Run the full defect-detection pipeline.

        Args:
            image:      BGR image (preprocessed 224×224 ROI or raw).
            mask:       Optional pre-computed mango mask (255=mango).
                        **Recommended**: pass the HSV mask created by
                        ``create_mango_hsv_mask`` on the *raw* ROI,
                        resized to 224×224.  If ``None``, a mask is
                        computed from ``image`` directly (less reliable
                        on preprocessed images).
            save_debug: Save intermediate debug images.
            debug_path: Directory for debug images.

        Returns:
            DefectAnalysis dataclass.
        """
        import time
        t0 = time.time()
        img_area = image.shape[0] * image.shape[1]

        # 1. Mask — use supplied mask or compute from image
        if mask is not None:
            mango_mask = mask
            # Resize if dimensions don't match
            if mango_mask.shape[:2] != image.shape[:2]:
                mango_mask = cv2.resize(mango_mask,
                                        (image.shape[1], image.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
                _, mango_mask = cv2.threshold(mango_mask, 127, 255,
                                              cv2.THRESH_BINARY)
        else:
            mango_mask = self.create_mango_mask(image)
        mango_area = float(cv2.countNonZero(mango_mask))
        if mango_area == 0:
            mango_area = float(img_area)
            mango_mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

        # 2. Preprocess (colour-space conversions; skips CLAHE if configured)
        prep = self.preprocess_image(image)

        # 3. Detect
        dark  = self.detect_dark_spots(prep['enhanced'], mango_mask, prep['hsv'])
        brown = self.detect_brown_spots(prep['hsv'], mango_mask)
        all_defects = dark + brown

        # 4. Metrics
        total_area = sum(d.area for d in all_defects)
        defect_pct = (total_area / mango_area) * 100.0

        # 5. Colour uniformity
        uniformity = self.analyze_color_uniformity(image, mango_mask)

        # 6. Scores
        quality = self.calculate_surface_quality_score(all_defects)
        impact  = self.assess_export_grade_impact(defect_pct, all_defects, img_area)

        elapsed = time.time() - t0

        if save_debug and debug_path:
            self._save_debug(image, prep, mango_mask, all_defects, debug_path)

        return DefectAnalysis(
            total_defect_area=total_area,
            total_defect_percentage=defect_pct,
            mango_area=mango_area,
            defect_count=len(all_defects),
            dark_spot_count=len(dark),
            brown_spot_count=len(brown),
            defect_regions=all_defects,
            color_uniformity_score=uniformity,
            surface_quality_score=quality,
            export_grade_impact=impact,
            processing_time=elapsed,
        )

    # ──────────────────────────────────────────────────────────────────────
    #  Visualisation
    # ──────────────────────────────────────────────────────────────────────
    def visualize_defects(self, image: np.ndarray,
                          analysis: DefectAnalysis) -> np.ndarray:
        """Draw annotated defect overlay on a copy of the image."""
        vis = image.copy()
        h_img, w_img = vis.shape[:2]

        # Scale font & line thickness to image size
        scale = max(0.3, min(h_img, w_img) / 400)
        thick = max(1, int(scale * 2))

        colors = {'dark_spot': (0, 0, 255), 'brown_spot': (0, 165, 255)}

        for d in analysis.defect_regions:
            c = colors.get(d.type, (255, 255, 255))
            cv2.drawContours(vis, [d.contour], -1, c, thick)
            if d.severity in ('moderate', 'severe'):
                x, y, w, h = d.bounding_box
                cv2.rectangle(vis, (x, y), (x + w, y + h), c,
                              thick + 1 if d.severity == 'severe' else thick)
            label = 'dark' if d.type == 'dark_spot' else 'brown'
            cv2.putText(vis, label,
                        (d.center[0] - 12, d.center[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, scale * 0.6, c, thick)

        lines = [
            f"Dark: {analysis.dark_spot_count}  Brown: {analysis.brown_spot_count}",
            f"Defect: {analysis.total_defect_percentage:.1f}%",
            f"Uniformity: {analysis.color_uniformity_score:.0f}/100",
            f"Quality: {analysis.surface_quality_score:.0f}/100",
            f"Impact: {analysis.export_grade_impact}",
        ]
        fs = scale * 0.55
        y = int(18 * scale)
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX,
                                          fs, thick)
            cv2.rectangle(vis, (4, y - th - 2), (8 + tw, y + 2), (0, 0, 0), -1)
            cv2.putText(vis, line, (6, y),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 0), thick)
            y += int(22 * scale)

        return vis

    # ──────────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────────────────────
    def _classify_severity(self, area: float, mean_int: float,
                           img_area: int) -> str:
        sev_a = self._scale(self.config['sev_severe_area'], img_area)
        mod_a = self._scale(self.config['sev_moderate_area'], img_area)
        if area > sev_a or mean_int < self.config['sev_severe_intensity']:
            return 'severe'
        if area > mod_a or mean_int < self.config['sev_moderate_intensity']:
            return 'moderate'
        return 'minor'

    @staticmethod
    def _spot_confidence(contour: np.ndarray, area: float) -> float:
        peri = cv2.arcLength(contour, True)
        if peri > 0:
            circ = 4 * np.pi * area / (peri * peri)
            return min(1.0, 0.3 + circ * 0.7)
        return 0.5

    def _save_debug(self, orig, prep, mask, defects, path):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p / "01_original.jpg"), orig)
        cv2.imwrite(str(p / "02_mango_mask.jpg"), mask)
        cv2.imwrite(str(p / "03_enhanced.jpg"), prep['enhanced'])

        dk = np.zeros(mask.shape, np.uint8)
        br = np.zeros(mask.shape, np.uint8)
        for d in defects:
            target = dk if d.type == 'dark_spot' else br
            cv2.drawContours(target, [d.contour], -1, 255, -1)
        cv2.imwrite(str(p / "04_dark_spots.jpg"), dk)
        cv2.imwrite(str(p / "05_brown_spots.jpg"), br)


__all__ = ['MangoDefectDetector', 'DefectAnalysis', 'DefectRegion']
