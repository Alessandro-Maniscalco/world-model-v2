# Checkpoints On GitHub

This repo tracks only `saved_checkpoints/vae_pickplace_z32_8x/best.pt`, and it is stored with Git LFS.

To replace that checkpoint with a newer `best.pt`:

1. Install Git LFS on the machine that will push the change.
2. Run `git lfs install` once in your shell.
3. Overwrite `saved_checkpoints/vae_pickplace_z32_8x/best.pt` with the new file.
4. Run `git add saved_checkpoints/vae_pickplace_z32_8x/best.pt .gitattributes .gitignore saved_checkpoints/checkpoints_on_github.md`.
5. Commit and push.

To track a different checkpoint path instead, update both `.gitattributes` and the `saved_checkpoints` exceptions in `.gitignore`.
