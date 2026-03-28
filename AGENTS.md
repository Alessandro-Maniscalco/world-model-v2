<INSTRUCTIONS>
## Docstrings
- Every source file must start with a top of the file module docstring describing the module’s purpose.
- Exclude vendored, generated, or third-party code.
- Every function should have a concise docstring describing its purpose.
## Testing
- Every new source file containing logic should have a corresponding test file in the `tests/` directory.
- Exception: lightweight smoke-check scripts under `scripts/check/` do not require a corresponding pytest file when they are only intended for manual execution.
- Use `pytest` for testing.
- If the logic depends on external models or heavy resources (like VAEs), use mocks to keep tests fast and deterministic.
- When it makes sense, run the command you are telling me to run, check whether if it runs correctly for the amount of time you feel reasonable, fix it if it fails, and repeat until there is no error.
- For bugs visible only in generated artifacts or end-to-end outputs, do not stop at code inspection or unit tests. Reproduce the exact failing artifact with the real command or closest local reproduction, trace the full pipeline stage by stage, and verify the repaired artifact after the fix.
- For video or temporal-model bugs, explicitly check all relevant boundaries that can silently change frame counts or alignment, including raw input frames, latent-time shapes, decoded frame counts, and exported video frame counts.
## Environment
- Always run Python and test commands inside the repo virtualenv at `.venv`.
- Before running Python tools, execute: `source .venv/bin/activate`.
##Design decisions
- Every major design choice must be written in the chat (what changed and why).
- After changing code, review whether any older code has become unused and delete it when it is no longer needed.
- Do not create `docs/plans/` design or implementation markdown files for lightweight manual smoke-check scripts under `scripts/check/`.
##Dependencies
- If a new library must be installed, add it to requirements.txt in the same change.
</INSTRUCTIONS>
