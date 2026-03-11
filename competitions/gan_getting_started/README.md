# GAN Getting Started

Workspace for Kaggle competition:

- competition slug: `gan-getting-started`
- title: `I'm Something of a Painter Myself`
- task: generate Monet-style images and submit as `images.zip`

## Layout

- `scripts/download_data.py`: download and extract competition files
- `scripts/build_submission.py`: build a valid baseline `images.zip` from local jpgs
- `scripts/train_model.py`: wrapper that creates baseline submission artifact
- `tests/`: unit tests for downloader and submission builder

## Quick Start

```bash
python competitions/gan_getting_started/scripts/download_data.py
python competitions/gan_getting_started/scripts/train_model.py
```

## Notes

- Baseline submission script uses raw `photo_jpg` images as placeholders and renames
  them to `00000.jpg`, `00001.jpg`, ... inside `images.zip`.
- This baseline is for pipeline validation; it is not competitive.
- This competition is notebook-submission-only on Kaggle:
  - local `kaggle competitions submit` is rejected
  - run a notebook on Kaggle that writes `/kaggle/working/images.zip` and submit that version
