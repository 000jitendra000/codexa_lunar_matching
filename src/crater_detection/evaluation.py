"""
src/crater_detection/evaluation.py

Evaluation infrastructure for crater detection.
Implements one-to-one greedy bipartite matching using Intersection over Union (IoU)
to calculate precision, recall, F1-score, and mean IoU against ground truth crater candidates.

NOTE: This is evaluation infrastructure for future labeled dataset testing.
Do not use this to fabricate lunar performance metrics without real ground truth.
"""

import logging
from typing import List, Dict, Any, Tuple
from src.crater_detection.types import CraterCandidate
from src.crater_detection.postprocess import compute_crater_iou

logger = logging.getLogger(__name__)


def compute_crater_detection_metrics(ground_truth: List[CraterCandidate],
                                     predictions: List[CraterCandidate],
                                     iou_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Compute detection metrics comparing predictions to ground truth crater candidates.

    Matching Criterion (One-to-One Greedy Matching):
    -----------------------------------------------
    1. Compute all pairwise IoU values between ground truth and predicted bounding boxes.
    2. Sort all candidate pairs (gt_idx, pred_idx) descending by IoU.
    3. Greedily match pairs where IoU >= iou_threshold:
       - Each ground-truth crater is matched to at most one predicted crater.
       - Each predicted crater is matched to at most one ground-truth crater.
    4. Counting:
       - True Positives (TP): Number of matched (gt, pred) pairs with IoU >= iou_threshold.
       - False Positives (FP): Predicted craters that were not matched to any ground-truth crater.
       - False Negatives (FN): Ground-truth craters that were not matched by any prediction.
    5. Mean IoU:
       - Average IoU across all True Positive matched pairs (0.0 if TP == 0).

    Args:
        ground_truth: List of ground-truth CraterCandidate objects.
        predictions: List of predicted CraterCandidate objects.
        iou_threshold: Minimum IoU required to consider a prediction as a match (default: 0.5).

    Returns:
        Dict containing precision, recall, f1_score, true_positives, false_positives,
        false_negatives, mean_iou, iou_threshold, gt_count, and pred_count.
    """
    n_gt = len(ground_truth)
    n_pred = len(predictions)

    # Edge cases: empty inputs
    if n_gt == 0 and n_pred == 0:
        return {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
            "mean_iou": 0.0,
            "iou_threshold": iou_threshold,
            "gt_count": 0,
            "pred_count": 0,
        }

    if n_gt == 0 and n_pred > 0:
        return {
            "true_positives": 0,
            "false_positives": n_pred,
            "false_negatives": 0,
            "precision": 0.0,
            "recall": 1.0,
            "f1_score": 0.0,
            "mean_iou": 0.0,
            "iou_threshold": iou_threshold,
            "gt_count": 0,
            "pred_count": n_pred,
        }

    if n_gt > 0 and n_pred == 0:
        return {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": n_gt,
            "precision": 1.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "mean_iou": 0.0,
            "iou_threshold": iou_threshold,
            "gt_count": n_gt,
            "pred_count": 0,
        }

    # Compute all pairwise IoUs
    pairs: List[Tuple[float, int, int]] = []
    for g_idx, gt in enumerate(ground_truth):
        for p_idx, pred in enumerate(predictions):
            iou = compute_crater_iou(gt, pred)
            if iou >= iou_threshold:
                pairs.append((iou, g_idx, p_idx))

    # Sort descending by IoU
    pairs.sort(key=lambda item: item[0], reverse=True)

    matched_gt = set()
    matched_pred = set()
    matched_ious: List[float] = []

    for iou, g_idx, p_idx in pairs:
        if g_idx not in matched_gt and p_idx not in matched_pred:
            matched_gt.add(g_idx)
            matched_pred.add(p_idx)
            matched_ious.append(iou)

    tp = len(matched_gt)
    fp = n_pred - tp
    fn = n_gt - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = float(sum(matched_ious) / len(matched_ious)) if matched_ious else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "mean_iou": round(mean_iou, 4),
        "iou_threshold": iou_threshold,
        "gt_count": n_gt,
        "pred_count": n_pred,
    }
