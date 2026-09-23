import logging
import cv2
from src.matching.hybrid_matcher import HybridMatcher

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    image_a = cv2.imread("data/raw/image_a.png", cv2.IMREAD_GRAYSCALE)
    image_b = cv2.imread("data/raw/image_b.png", cv2.IMREAD_GRAYSCALE)

    if image_a is None or image_b is None:
        raise RuntimeError("Could not load one or both images.")

    print("Image A:", image_a.shape)
    print("Image B:", image_b.shape)

    print("\nInitializing HybridMatcher...")
    matcher = HybridMatcher()

    print("Running hybrid matcher...")
    result = matcher.match(image_a, image_b)

    print("\n" + "=" * 50)
    print("REAL LUNAR IMAGE TEST")
    print("=" * 50)

    print("Matched:", result.matched)
    print("Correspondences:", len(result.correspondences))
    print("Inliers:", result.num_inliers)
    print("Inlier ratio:", result.inlier_ratio)
    print("Confidence:", result.confidence)

    if result.transform is not None:
        print("\nTransform:")
        print("Scale:", result.transform.scale)
        print("Rotation:", result.transform.rotation_deg)
        print("Translation:", result.transform.translation_x, result.transform.translation_y)

    print("\nMetadata:")
    print(result.metadata)

if __name__ == "__main__":
    main()
