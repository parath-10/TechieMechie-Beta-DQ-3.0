# Deploying NEXUS on Railway + Supabase

## 1. Create the Supabase project
1. Go to https://supabase.com, sign in, then **New project**.
2. Pick a region close to your users (for India: **Mumbai, ap-south-1**). Save the database password somewhere safe.
3. Wait until the project says it is ready (about 2 minutes).

## 2. Create the tables, functions and storage bucket
1. In the project, open **SQL Editor** and click **New query**.
2. Paste the whole of `schema.sql` and click **Run**. It should end with "Success. No rows returned".
3. Check: **Table Editor** shows `products, ads, posts, orders, monthly_metrics, connections, actions, app_state`,
   and **Storage** shows a public bucket called `ad_creatives`.

## 3. Collect the keys
1. **Project Settings -> API** (or **Data API**): copy the **Project URL** -> this is `SUPABASE_URL`.
2. **Project Settings -> API Keys**: copy the **service_role** key (or the newer **Secret key**, starting `sb_secret_`)
   -> this is `SUPABASE_SERVICE_ROLE_KEY`. If the newer key gives an authentication error, use the
   `service_role` key from the **Legacy API keys** tab.
   Never put this key in index.html or share it: it can read and change everything.
3. Make the encryption key for saved platform tokens:
   `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
   -> this is `NEXUS_SECRET_KEY`. Keep a copy: if it changes, saved connections must be reconnected.

## 4. (Optional) Test on your computer first
1. Copy `.env.example` to `.env` and fill in the values from step 3 plus `GEMINI_API_KEY`.
2. `pip install -r requirements.txt`
3. `uvicorn main:app --reload --port 8000`, then open http://localhost:8000
4. The first start creates the sample products in Supabase (you will see them in Table Editor).

## 5. Put the code on GitHub
1. Create a new **private** repository.
2. Upload every file from this folder **except** `.env` (the `.gitignore` already leaves it out).

## 6. Deploy on Railway
1. Go to https://railway.app -> **New Project -> Deploy from GitHub repo** -> pick your repository.
2. Open the service -> **Variables** -> **Raw Editor**, and paste:
   ```
   SUPABASE_URL=...
   SUPABASE_SERVICE_ROLE_KEY=...
   NEXUS_SECRET_KEY=...
   GEMINI_API_KEY=...
   APP_TIMEZONE=Asia/Kolkata
   ```
   (Add any optional settings from `.env.example` the same way.)
3. **Settings -> Deploy**: the start command comes from the `Procfile`. If Railway does not pick it up,
   set **Custom Start Command** to `uvicorn main:app --host 0.0.0.0 --port $PORT`.
4. **Settings -> Deploy -> Healthcheck Path**: `/health` (optional but recommended).
5. **Settings -> Networking -> Generate Domain**. Open the address: the dashboard loads from `/`.

## 7. Check it works
- Products, Inventory and the monthly graph show data.
- Connected Platforms -> "Connect all with demo accounts" -> rows appear in the `connections` table
  (tokens appear only as encrypted text starting `gAAAA`).
- Post an ad with a picture -> the file appears in Storage -> `ad_creatives`.
- Restart the Railway service: everything is still there.

## Useful notes
- **Start again with fresh sample data** (SQL Editor):
  `truncate products, orders, connections, actions, app_state cascade;` then restart the Railway service.
- **No sample data at all**: set `SEED_SAMPLE_DATA=0` before the first start.
- **Video size**: Supabase Free allows files up to 50 MB. On Pro, raise the bucket limit
  (Storage -> ad_creatives -> Edit) and set `MAX_VIDEO_MB=100`.
- **Old local connections** (`connections.json`) are not copied over: just reconnect the platforms once.
