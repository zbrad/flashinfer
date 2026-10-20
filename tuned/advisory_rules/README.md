# tuned/advisory_rules

Upstream's agent rule files (CLAUDE.md, AGENTS.md, .claude/ skills and similar), moved here with `git mv` so that coding
agents do not load them by default. They are kept exactly as upstream wrote them, at the same relative path they had, so
a reader can still consult them and a merge from upstream can still carry upstream edits through the rename.

They are reference material, not instructions for this fork: the owner's own rules are authoritative. When one of them
informs work, name it as upstream's. Do not move them back. If a merge from upstream brings an agent rule file back to a default
path, `git mv` it here (tuned-common's upstream-sync.sh is to do this; until it does, do it by hand).
