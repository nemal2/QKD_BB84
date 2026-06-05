# Deployment Guide — BB84 Simulator + Survey System

The app is a single Streamlit process. The integrated survey system stores data
through a **pluggable storage layer**:

| Where it runs | Storage used | Why |
|---|---|---|
| Your laptop / a VPS | **SQLite** file at `survey_data/qkd_survey.db` (automatic) | Zero setup, persists on a real disk |
| Heroku / DigitalOcean App Platform | **Postgres** via `DATABASE_URL` (automatic when set) | Their filesystem is *ephemeral* — a SQLite file is wiped on every restart/redeploy |

> ⚠️ **Read this once.** Heroku dynos and DigitalOcean's App Platform reset their
> filesystem on each restart. If you deploy there **without** a `DATABASE_URL`, the
> app still runs but every survey response is lost on the next reboot. Always
> provision a managed Postgres on those platforms — then it "just works", no code
> change.

---

## Configuration (env vars / secrets)

| Key | Required | Purpose |
|---|---|---|
| `DATABASE_URL` | Prod only | Managed Postgres URL. Unset ⇒ local SQLite. `postgres://` is auto-normalised. |
| `SURVEY_ADMIN_PASSWORD` | To use admin | Unlocks the dashboard at `/?admin=1`. |
| `SURVEY_ENABLED` | No | `0` disables the survey gate (bare simulator). Default `1`. |

Set these as **environment variables** (Heroku/DO) or, for local/Streamlit Cloud,
copy `.streamlit/secrets.toml.example` → `.streamlit/secrets.toml` and fill it in.

---

## Optional: Google Sheets secondary mirror

The SQL database is always the source of truth. You can *additionally* mirror data
into a Google Sheet — a live, browser-viewable backup. It's entirely optional; skip
it and nothing changes.

1. In Google Cloud, create a **service account**, enable the **Google Sheets API**,
   and download the service account's **JSON key**.
2. Create a Google Sheet, then **Share** it (Editor) with the service account's
   `client_email`.
3. Provide credentials:
   - **Local / Streamlit Cloud:** paste `gsheets_id` and the `[gcp_service_account]`
     block into `.streamlit/secrets.toml` (see the example file).
   - **Heroku / DigitalOcean:** set env vars `GSHEETS_ID` and
     `GCP_SERVICE_ACCOUNT_JSON` (the JSON key as a single-line string).

Once configured, responses and activity append live; participant rows update on each
stage change. The admin dashboard gains a **“Sync all to Google Sheets”** button that
rebuilds every tab from the database. If Sheets is ever unreachable, the app keeps
working — the mirror is best-effort and never blocks a participant.

---

## Local development

```bash
pip install -r requirements.txt
export SURVEY_ADMIN_PASSWORD="dev-password"     # optional, to try the dashboard
streamlit run qkd_app.py
```

Data lands in `survey_data/qkd_survey.db` (gitignored). Open the admin dashboard at
`http://localhost:8501/?admin=1`.

---

## Heroku

```bash
heroku create your-qkd-app
heroku addons:create heroku-postgresql:essential-0   # sets DATABASE_URL for you
heroku config:set SURVEY_ADMIN_PASSWORD="a-strong-password"
git push heroku <your-branch>:main
heroku open
```

`Procfile` and `.python-version` are already in the repo. `psycopg2-binary` is in
`requirements.txt`, so the Postgres driver is installed automatically.

---

## DigitalOcean App Platform

1. **Create → App** from your Git repo.
2. Add a **Dev Database → PostgreSQL** (or attach a Managed Database). DO injects a
   `DATABASE_URL` automatically — the app picks it up.
3. **Settings → App-Level Environment Variables**: add `SURVEY_ADMIN_PASSWORD`.
4. **Run Command:**
   ```
   streamlit run qkd_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
   ```
5. Deploy.

> Prefer a **Droplet** (plain VM) instead? Then the disk is persistent and the
> default SQLite file is fine — no database add-on needed. Run it behind nginx +
> `systemd`, or in Docker with a mounted volume for `survey_data/`.

---

## After deploying

- **Participants:** share the app URL. Each visitor gets an anonymous code
  (`QKD-XXXX-XXXX`) kept in their URL, so a refresh resumes their progress.
- **Researcher:** open `/?admin=1`, log in, watch the pre/post comparison, and
  export `participants.csv` / `responses.csv` / `activity.csv` for analysis.
- **Edit the questions:** everything you ask lives in
  [`survey/questions.py`](survey/questions.py). Replace the `[PLACEHOLDER]` items;
  the database, scoring and exports adapt automatically. Don't rename a question
  `id` once real data exists.
