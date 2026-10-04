# Model card and licensing

## Decision

The project will target the upstream [YOLOX source repository](https://github.com/Megvii-BaseDetection/YOLOX), whose repository license is Apache-2.0. This avoids the AGPL-3.0 obligations of the Ultralytics option for the serving code path.

## Weights status: blocked pending verification

No pretrained checkpoint is downloaded or redistributed by this repository. The upstream
repository's Apache-2.0 source license does **not** establish the license for its hosted
weights. Before selecting a checkpoint, record its exact URL, SHA-256, license, provenance,
and commercial-use terms here. If these cannot be verified from the publisher, use a different
model/weights release with explicit permissive terms.

## Intended use and limits

Object detection for defensive, consent-respecting applications. This is not validated for
biometric identification, safety-critical decisions, or adversarial inputs. Accuracy and bias
will be reported only after a licensed evaluation set and reproducible script are in place.
