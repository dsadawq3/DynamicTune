# Naming and migration boundary

The directory and Python import path remain unchanged in this pass so that
existing local artifacts and tests keep a stable boundary. `LATUNE` is a
candidate conceptual name meaning **Latent Tune**; it is not applied as a
package or directory rename here.

The implementation vocabulary is deliberately about latent charts, observed
trajectories, differential signatures, flow operators, and student-side
corrections. A future rename can change display metadata and the CLI entry
point through a compatibility layer without changing artifact schemas or
mathematical contracts.
