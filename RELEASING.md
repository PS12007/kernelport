# Releasing kernelport to PyPI

Publishing is done by `.github/workflows/release.yml` when a `v*` tag is pushed. It builds the
sdist and wheel, checks that the tag matches `version` in `pyproject.toml`, and uploads with PyPI
**trusted publishing** (no API token is stored anywhere).

## One-time setup (pypi.org)

1. Sign in to https://pypi.org (create an account and turn on 2FA if needed).
2. Go to https://pypi.org/manage/account/publishing/ and, under **Add a new pending publisher**,
   choose GitHub and fill in:

   | field | value |
   |---|---|
   | PyPI Project Name | `kernelport` |
   | Owner | `PS12007` |
   | Repository name | `kernelport` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   A *pending* publisher reserves the name until the first upload creates the project.
3. On GitHub: repo **Settings → Environments → New environment**, name it `pypi`. Optionally add
   yourself as a required reviewer, so every release waits for one click of approval.

## Each release

1. Set the version in `pyproject.toml`, e.g. `version = "0.1.0"`, and commit.
2. Tag and push:

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

3. Watch **Actions → Release**. If you added a reviewer to the `pypi` environment, approve the run.
4. Check https://pypi.org/project/kernelport/ and try a clean install:

   ```bash
   python -m venv /tmp/kp && /tmp/kp/bin/pip install kernelport && /tmp/kp/bin/kernelport doctor
   ```

5. After the first release, switch the install lines in `README.md` and `docs/installation.md`
   from `git+https://github.com/PS12007/kernelport` to plain `kernelport`, e.g.
   `pip install "kernelport[cuda,triton,cpu]"`, and set the version to the next `.dev0`.

A version can be uploaded only once. If something is wrong after upload, fix it and release the
next patch version (`0.1.1`); `pip` ignores yanked versions unless they are pinned exactly.
