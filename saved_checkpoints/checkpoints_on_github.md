# Checkpoints On GitHub

This repo tracks only `saved_checkpoints/github/vae_pickplace_z32_8x.pt`, and it is stored with Git LFS.

The previous checkpoint that used to live at that path has been archived locally under
`saved_checkpoints/old/old_vae_pickplace_z32_8x/`.
The current file was copied from
`outputs/so101_episode0_full_ae_resume_from_f113_137_crop_120x160_4xspatial_z32/checkpoints/best.pt`.

To replace that checkpoint with a newer `.pt` file:

1. Install Git LFS on the machine that will push the change.
2. Run `git lfs install` once in your shell.
3. Overwrite `saved_checkpoints/github/vae_pickplace_z32_8x.pt` with the new file.
4. Run `git add saved_checkpoints/github/vae_pickplace_z32_8x.pt saved_checkpoints/checkpoints_on_github.md`.
5. Commit and push.

To track a different checkpoint path instead, update both `.gitattributes` and the `saved_checkpoints` exceptions in `.gitignore`.
