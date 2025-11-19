"""
ArcFace pipeline does not require a separate Siamese network.
This script remains for compatibility and simply documents the change.
"""

def main():
    print("\n" + "=" * 60)
    print("ARC FACE PIPELINE")
    print("=" * 60)
    print("\nThe current version of the attendance system uses ArcFace embeddings")
    print("with cosine similarity. No additional Siamese training is required.")
    print("\nSteps to improve accuracy:")
    print("  1. Capture high-quality images for each person (10+ per identity).")
    print("  2. Use the GUI/CLI to rebuild embeddings if you change the model.")
    print("  3. Adjust the similarity threshold in the GUI to tune sensitivity.")
    print("\nThat's it! Re-run the GUI or camera scripts to test recognition.")


if __name__ == "__main__":
    main()
