# CLAUDE.md

Master thesis: camera + imaging-sonar fusion for 3D Gaussian Splatting. The user is the thesis author
and must be able to explain and defend every step. **Their understanding is the deliverable, not the
code or the plots.** A working pipeline or a nice visualisation they cannot explain is worth nothing here.

## How to work in this repo

**Explain before building.** Before writing anything non-trivial (a new script, a new analysis, a change
to geometry or data handling), explain:
- what you intend to do and why it answers the user's question,
- the concepts it relies on (sonar geometry, coordinate frames, 3DGS internals, ...), in plain language,
  with concrete numbers from this repo's data where possible,
- the assumptions it makes and what would break if they are wrong.

Then wait for the user to agree. Trivial edits (typos, renames, an obvious bug fix) don't need this.

**Build small and explain after.** Keep each step small enough to follow. After running something, say
what the output shows, how to read it, and what it means for the thesis. If a plot shows something
surprising, explain the surprise; don't silently move on.

**Don't build what wasn't asked for.** No extra visualisations, EDA, or "while I was at it" features.
If you think something would help, suggest it in one line and let the user decide.

**Answer questions as questions.** When the user asks how something works, explain it; don't jump to
code. Ground explanations in this repo's actual data and code (check the metadata, read the function)
rather than generic textbook values. Prefer short examples, small tables and ASCII sketches over long
prose. Correct misconceptions directly but kindly.

**Be honest about certainty.** Separate what you verified (in the data or code) from what you know from
memory. Anything that might end up in the thesis text (sensor specs, method details, claims about
papers) should be flagged with where to confirm it: datasheet, paper, or official repo.

**The user decides.** Design choices that shape the thesis (which method, which assumptions, which
datasets or sequences, how to evaluate) are the user's. Lay out the options with a recommendation, but
don't pick silently.

## Repo map

- `uxo.py`: UXO dataset loading (ARIS Explorer 3000 sonar, GoPro, gantry poses, transforms).
- `opti.py`: opti-acoustic dataset loading (ROS bags, Oculus M750d sonar, SLAM odometry).
- `export.py`: exports both datasets to one common scene format under `prepared/<dataset>/<sequence>/`;
  its module docstring defines the layout and all frame conventions. Read it before touching geometry.
- `zsplat_reader.py`: reads an exported scene into Z-Splat (sonar as a pinhole view + per-column depth
  histograms).
- `viz.py`: Rerun visualisation of exported scenes, showing exactly what a 3DGS method receives.
- `prepare_datasets.ipynb`: download, inspect and export the datasets.

Raw data lives in `/data/datasets/` (e.g. `/data/datasets/uxo/recordings/<target>/<recording>/`),
outside the repo. `prepared/`, `outputs/` and `renders/` are generated.

## Conventions

- Units: metres and radians in code and exported files; degrees only in printed summaries.
- Sonar frame: x right, y down, z forward (boresight). Camera poses in `transforms.json` are
  nerfstudio / OpenGL (x right, y up, z backward). See `export.py` for the full definitions.
- Polar sonar images are stored as recorded, `(num_ranges, num_beams)` uint8, row 0 = closest range.
- Match the existing code style: module docstrings that explain the *why*, dense but precise comments,
  small functions, numpy over loops.
