# Artifact governance

This artifact separates frozen manuscript evidence from future living updates.

For a correction or new record, open an issue or pull request containing the
bibliographic identifier, claimed threat and subfield, probe manipulation,
outcome, effect scale, model and version, access date, decoding settings,
publication status, license, and the exact source passage or released output
that supports the record. Null and contradictory effects are accepted under the
same schema as dramatic effects.

A maintainer checks identifier resolution, source support, scope, license, and
schema validity. Changes to a published value require a regression-test update,
a note in the release changelog, and a new versioned release. Existing releases
remain immutable. Corrections never silently overwrite the evidence used by a
paper version.

Coverage coding requires representative evidence for each cell. Future
re-coding should retain both independent coder outputs, prompts, model versions,
decoding settings, and adjudication. The current release contains only the
final ledger, so its historical agreement statistic cannot be reconstructed.
