"""Irina-RL: the self-improvement lab around the Odyssey harness.

Everything that makes the harness *learn* lives here, and nothing here edits
the harness. `provider_local.py` gives the scan the face of a drop-in
provider; `traj.py` converts a neutral transcript into a trainable example;
`grpo.py` and `offline.py` do the two learning loops; `teacher.py` is the
DeepSeek judge; `insight_pool.py` banks successful runs; `colab_gradio.py` is
the UI. See `PLAN.md` at the repo root for the roadmap.
"""