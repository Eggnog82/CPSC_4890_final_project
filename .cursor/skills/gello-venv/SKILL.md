---
name: gello-venv
description: >-
  Uses the Python virtualenv at gello_software/.venv for GELLO and related
  installs/commands. Apply when the user mentions gello_software, GELLO,
  installing packages in the project venv, matplotlib, or running scripts under
  gello_software.
---

# GELLO software virtualenv

## Location

From the **repository root** (`final_project/`):

- Venv directory: **`gello_software/.venv`**
- Python: **`gello_software/.venv/bin/python`**
- Pip: **`gello_software/.venv/bin/pip`** (after `ensurepip` or a normal `python -m venv` creation)

## Installing packages

Prefer **`python -m pip`** so pip matches the venv interpreter:

```bash
cd gello_software
source .venv/bin/activate
python -m pip install -U pip
python -m pip install <package>
```

One-shot without activating:

```bash
gello_software/.venv/bin/python -m pip install matplotlib
```

## Running GELLO / scripts

```bash
cd gello_software
source .venv/bin/activate
python experiments/run_env.py ...
```

## If `pip` is missing inside `.venv`

Bootstrap pip, then install:

```bash
cd gello_software
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install -U pip
```

---

**Note:** The **lab2** stack may use a **separate** venv under `lab2/.venv` if present; use that only for `lab2` scripts, not for `gello_software`.
