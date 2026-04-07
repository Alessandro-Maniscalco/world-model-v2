## Goal

Trying to overfit on 6 images from the dataset.

## Plan

Train on frames 111-116 of episode 0.
Train all 200 frames of episode 0.
Train on 10 episodes of the same task.
Train on 10 episodes of 10 same tasks, validation on 11th episode. From now one test and validation are separate.
Train on all episode of same task.
Train on X episodes of all tasks.

The model will change as it might not be good. After chaning it, an overfit of the 6 frames will always be done.

## Stable finidngs
- vae_only and then dynamics_only works better than joint training, removed joint training

## Frames 111-116 of episode 0 with vae

Encode decode works with the same test and validation data but it diverges when images are horiontally flipped.

The dynamics model work for open rollout, best at step 800 and then it inroduces noise and gets worst.

200 frames to see if the vae works

I used all the episodes from the task single_grasp and the vae works well. There is some blurriness and loss of detail, for example the arm is not pointy. Stopped at 3000 steps but doesnt look like it was improving

The best run is here: outputs/minimal/ae_only_single_grasp_full_240x320_dim64_z16_k3e5_b4


