import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

import stgcn_variants

DETERMINISTIC_HINTS = ("does not have a deterministic implementation",
                       "deterministic")


def main() -> int:
    print("[driver] CUBLAS_WORKSPACE_CONFIG="
          f"{os.environ['CUBLAS_WORKSPACE_CONFIG']} "
          "use_deterministic_algorithms=True cudnn.benchmark=False (F7)")
    try:
        stgcn_variants.main()
    except RuntimeError as exc:
        msg = str(exc)
        if any(hint in msg for hint in DETERMINISTIC_HINTS):
            print("=" * 78)
            print("DETERMINISTIC-OP-FAILURE: a torch op has no deterministic "
                  "implementation on this device.")
            print("F7 pre-registered fallback -- rerun on CPU with 8 threads, "
                  "seed 42, no further discussion:")
            print("    python run_nested_variants.py "
                  "--variants base,V1,V2,V3,V4,V5,V6 "
                  "--device cpu --threads 8 --outdir gen/nested/variants")
            print("=" * 78)
            return 2
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
