"""Rebuild the deployment bundle using the canonical app training pipeline."""

import os


def main():
    os.environ["SURVIVAL_REBUILD_BUNDLE"] = "1"
    import survival_app

    bundle = survival_app._B
    for model in ("cox", "aft"):
        train, test = bundle[f"tr_{model}"], bundle[f"res_{model}"]
        print(f"{model}: train/test C-index={train['c_index']:.6f}/{test['c_index']:.6f}; "
              f"train/test IBS={train['ibs']:.6f}/{test['ibs']:.6f}")
    print(f"Saved: {survival_app._BUNDLE_PATH}")


if __name__ == "__main__":
    main()
