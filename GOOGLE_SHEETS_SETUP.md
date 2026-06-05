# Storing survey data in the cloud (Google Sheets) — step by step

## First, the mental model

Your survey data lives in **two layers**:

1. **Primary store — the database.** This is the real, complete record. When the app
   runs on your DigitalOcean droplet, the database file lives on that server — which
   *is* a cloud machine — so your data is already "in the cloud" and safe across
   restarts (a droplet has a persistent disk).

2. **Secondary store — Google Sheets (optional, what you're asking about).** A live
   *copy* in your Google Drive that you can open in a browser, filter, and share with
   your supervisor. The app code for this is already built; you just need to connect
   your Google account once.

> The Sheet is a **mirror, not the master.** If Google is ever unreachable, the app
> keeps working and nothing is lost — the database still has everything, and the
> admin "Sync all" button can rebuild the Sheet at any time.

---

## What YOU need to do (one-time, ~10 minutes)

You'll create a **service account** — a robot Google account the app logs in as — and
share one Google Sheet with it.

### Step 1 — Create a Google Cloud project
1. Go to <https://console.cloud.google.com> and sign in with your Google account.
2. Top bar → project dropdown → **New Project** → name it `qkd-survey` → **Create**.
3. Make sure the new project is selected (top bar).

### Step 2 — Turn on the Google Sheets API
1. Search bar → type **"Google Sheets API"** → open it → **Enable**.
2. (Optional but harmless) do the same for **"Google Drive API"**.

### Step 3 — Create the service account + key file
1. Left menu → **APIs & Services → Credentials**.
2. **+ Create credentials → Service account**. Name it `qkd-survey-bot` → **Create and continue** → **Done**.
3. Click the new service account → **Keys** tab → **Add key → Create new key → JSON → Create**.
4. A `.json` file downloads. **Keep it private — never commit it to git.**
5. Copy the service account's **email** (looks like
   `qkd-survey-bot@qkd-survey.iam.gserviceaccount.com`). You'll need it next.

### Step 4 — Create the Sheet and share it
1. Go to <https://sheets.new> → rename the sheet, e.g. **"QKD Survey Data"**.
2. From the URL, copy the **spreadsheet id** — the long part between `/d/` and `/edit`:
   `https://docs.google.com/spreadsheets/d/`**`THIS_IS_THE_ID`**`/edit`
3. Click **Share** → paste the service-account **email** from Step 3 → set it to
   **Editor** → untick "Notify people" → **Share**.
   *(This is the step people forget — the robot can only write to sheets you've shared
   with it.)*

### Step 5 — Give the credentials to the app
Open the downloaded JSON key and copy its fields into
`.streamlit/secrets.toml` (this file is gitignored, so it never reaches GitHub).
Use `.streamlit/secrets.toml.example` as the template:

```toml
# admin dashboard password (you chose this)
SURVEY_ADMIN_PASSWORD = "fyp2026QKD"

# the Sheet from Step 4 (id or full URL both work)
gsheets_id = "PASTE_THE_SPREADSHEET_ID_HERE"

# paste from the downloaded JSON key — keep the \n's inside private_key exactly
[gcp_service_account]
type = "service_account"
project_id = "qkd-survey"
private_key_id = "…"
private_key = "-----BEGIN PRIVATE KEY-----\n…\n-----END PRIVATE KEY-----\n"
client_email = "qkd-survey-bot@qkd-survey.iam.gserviceaccount.com"
client_id = "…"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "…"
universe_domain = "googleapis.com"
```

Each key under `[gcp_service_account]` is just the matching field copied from the JSON.

### Step 6 — Verify it works
1. Restart the app (`streamlit run qkd_app.py`).
2. Open `http://<your-app>/?admin=1`, log in with the password.
3. Scroll to **"Secondary store · Google Sheets"** → click **⟳ Sync all to Google Sheets**.
4. Refresh your Google Sheet — three tabs appear: **participants**, **responses**,
   **activity**. From now on, new responses also append live.

---

## Doing the same on your DigitalOcean droplet (137.184.98.155)

The droplet can't see the JSON file on your laptop, so put the credentials there too:

- **Easiest:** copy your finished `.streamlit/secrets.toml` to the server next to
  `qkd_app.py`:
  ```bash
  scp .streamlit/secrets.toml root@137.184.98.155:/path/to/QKD_BB84/.streamlit/secrets.toml
  ```
  then restart the app process on the server.

- **Or use environment variables** (instead of the TOML file): set
  `GSHEETS_ID`, `GCP_SERVICE_ACCOUNT_JSON` (the whole JSON as one line),
  and `SURVEY_ADMIN_PASSWORD`, then restart.

That's it — no code changes, ever. The app detects the credentials and starts mirroring.

---

## Security reminders
- The downloaded JSON key and `secrets.toml` are **like passwords** — never commit
  them, never paste them in chat/email. Both are already gitignored.
- If a key leaks, delete it in **Cloud Console → service account → Keys** and make a new one.
- Anyone with the Sheet link who you share it with can read responses — keep sharing
  limited to your research team.
