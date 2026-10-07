# TechieMechie-Beta-DQ-3.0
# Nexus: The Ad Decision Engine for D2C Brands

> **Find the leaks. Fuel the winners.**
> Built at **DataQuest 3.0** by **Team TechieMechie**

🔗 **Live demo:** https://techiemechiebetadq.up.railway.app/

---

## 1. What is this, in simple words?

Imagine you run a small online shop. You sell on Instagram, Amazon, Flipkart and your own website, and you pay for ads on several of them. Every platform shows its own numbers in its own place, so you can't easily answer simple questions like:

- *"Am I actually **making money** on this ad, or only making sales?"*
- *"Why did my sales suddenly drop?"*
- *"Am I still paying for ads on a product that is **almost out of stock**?"*

**Nexus puts all of that in one place.** It brings together your ads, sales and stock, works out your **real profit**, and tells you in plain English **what is going wrong and what to do next**. You approve the suggestion and Nexus applies it.

### The three problems it catches

| Problem | What it means | What Nexus does |
|---|---|---|
| **Ad fatigue** | People have seen your ad too many times, so fewer of them click | Warns you early so you can refresh the ad before money is wasted |
| **Stockout guard** | Your ad is running for a product that is nearly sold out | Flags it and suggests pausing or capping the ad until you restock |
| **Revenue ≠ profit** | An ad looks great (high ROAS) but the product costs more than it earns | Shows the real profit after product cost, ads, shipping and fees |

---

## 2. What does the website contain?

The site has two parts: a **public landing page** and a **private dashboard** (login needed).

### Landing page (`/`)
- An animated intro and the headline, with live counters (products tracked, days analysed, ad spend at risk this week)
- "All your channels in one clear picture"
- The three problems above, shown with small charts
- A side-by-side example: *"Same product. Very different story."*
- A 5-step "How it works" section
- The **login box** (owner login, plus a view-only demo account for judges and visitors)

### Dashboard (`/dashboard`)

| Group | Page | What you do there |
|---|---|---|
| **Insights** | **Overview** | A daily summary: key numbers, an AI tip, alerts and a chat assistant. It refreshes every 30 seconds |
| | **Problems to Watch** | Shows ad fatigue per ad and which products are about to run out of stock |
| | **Profit Check** | Profit for every ₹1 spent on ads (above 1.00 means you make money) |
| | **Platform Analysis** | Compares Instagram, Facebook, YouTube, Google, Amazon, Flipkart and TikTok |
| **Manage** | **Products** | Add, edit, pause or delete products (price, cost, stock) |
| | **Inventory** | See how many days of stock are left and restock with one click |
| | **Ads** | Start, pause, stop or change the budget of ads per platform. You can also **post a new ad** with a headline, text, a picture or video, a budget and a duration |
| **Workspace** | **Things To Do** | AI-suggested actions (pause an ad, restock, move budget). **Nothing happens until you approve it** |
| | **Connected Platforms** | Connect Instagram, Facebook, YouTube, Google, TikTok, Shopify, Amazon and Flipkart |
| | **Time Log & Monthly** | Month-by-month sales, views and reviews, plus a timeline of events |

There is also a light/dark theme toggle, a bell icon for AI alerts, and a **review analyser** that reads customer feedback from different platforms and summarises it.

### Two kinds of login

| Account | Can do |
|---|---|
| **Owner** | Everything: change products, ads, accounts and settings |
| **Demo** (for judges and visitors) | Look at everything and use the chat and review analyser, but **cannot change** anything |

Demo login: `manager@nexus.com` / `nexus@123`

---

## 3. How it works (the working structure)

### The big picture

```
   Browser (index.html, dashboard.html)
        │   HTML + Tailwind + Alpine.js + Chart.js
        ▼
   FastAPI backend  (main.py)          ← login check on every request
        │
        ├── ads.py          products, ads, orders, monthly history, simulation
        ├── connectors.py   platform connections, encrypted tokens
        ├── ai_engine.py    AI insights, chat, review analysis
        ├── auth.py         login and signed session cookie
        └── db.py           one shared database client
        │
        ▼
   Supabase  (PostgreSQL database + file storage for ad pictures and videos)
        ▲
   Hosted on Railway    ·    AI by Groq (Gemini as backup)
```

### The 5-step decision flow

1. **Bring the data together.** Ad spend, orders, stock and product margins go into one set of tables.
2. **Spot unusual changes.** Sudden shifts in costs, clicks or sales are flagged.
3. **Find the root cause.** Stock, ad cost, click rate and margin are checked to explain *why* performance moved.
4. **Rank what to do.** Suggested actions are ordered by expected **profit**, not just revenue.
5. **Approve and learn.** You approve with one click, and the result is tracked against what was predicted.

### What happens when you use the dashboard

1. You **log in**. The server checks your password and gives you a signed, HttpOnly cookie. Every page and API call is checked against it, and the demo account is blocked from making changes.
2. The dashboard asks the backend for products, ads, orders and monthly numbers. The backend reads them from **Supabase** and works out days of stock left, profit per unit, profit after ads and so on.
3. Orders come in through an **order sync**. In this demo they are **simulated**, and time is sped up (1 real minute = 1 hour of orders) so you can watch numbers change. Each sync updates stock, month totals, per-ad results, orders and reviews in **one database transaction**, so nothing is half-saved.
4. When you connect a platform, its keys are **encrypted (Fernet/AES)** before they are saved. The browser only ever sees the last 4 characters.
5. When you post an ad, the picture or video is uploaded to **Supabase Storage**. The server checks the format rules first (for example, a Reel must be vertical and 90 seconds or less, and Flipkart takes images only).

### How the AI part works

- **One AI request creates everything at once**: the opening tip, suggestions, forecast, alerts, timeline and diagnosis. The result is saved in the database and reused by every page, tab and restart. A new request happens only when you press **Refresh AI**.
- **If the AI is down or its free limit is reached**, Nexus works out the insights from your live numbers using plain rules, so the dashboard always shows something useful.
- The **chat assistant** and **review analyser** call the AI only when you ask, sending a small summary of your data.
- The AI is told to use only the data given and never invent numbers.

### Database tables

| Table | Holds |
|---|---|
| `products` | Name, price, cost, stock, units sold, status |
| `ads` | One row per product per platform: status, daily budget |
| `posts` | Every ad created with "Post a new ad": creative, views, clicks, orders, spend |
| `orders` | The latest 300 synced orders |
| `monthly_metrics` | Units, revenue, views, clicks, spend, reviews per product, platform and month |
| `connections` | Connected platforms (secrets stored encrypted) |
| `actions` | Suggested actions waiting for approval |
| `app_state` | Shared values such as the sync clock and the saved AI insights |

Row Level Security is switched on, so only the backend can read or write. The public key sees nothing.

### Demo mode vs live mode

This is a hackathon prototype, so by default:
- Platform connections accept **demo accounts** ("Connect all with demo accounts" button), and ad numbers and orders are **simulated**.
- Set `LIVE_PLATFORM_APIS=1` and "Test connection" makes a real read-only call for Facebook, Instagram, YouTube and Shopify. Posting ads and pulling orders through the real platform APIs still needs approved developer apps, so the hooks are left as TODO.

---

## 4. Tech stack

| Layer | Tools |
|---|---|
| Frontend | HTML, Tailwind CSS, Alpine.js, Chart.js, hand-drawn SVG charts |
| Backend | Python 3.12, FastAPI, Uvicorn, Pydantic |
| Database and files | Supabase (PostgreSQL + Storage) |
| AI | Groq (`openai/gpt-oss-120b`) with Gemini as backup, through the OpenAI SDK |
| Security | Fernet encryption for tokens, HMAC-signed session cookie, login rate limiting |
| Hosting | Railway |

## 5. Project files

```
├── main.py            API routes, login gate, page serving
├── ads.py             products, ads, order simulation, monthly history
├── ai_engine.py       AI insights, chat, review analysis, rule-based fallback
├── connectors.py      the 8 platforms, encrypted token storage, connection tests
├── auth.py            owner/demo login, signed cookies
├── db.py              Supabase client and helper functions
├── schema.sql         tables, views, functions, security and storage bucket
├── index.html         landing page and login
├── dashboard.html     the dashboard
├── requirements.txt   Python packages
├── Procfile           start command for Railway
└── DEPLOY.md          step-by-step deployment guide
```

---

## 6. Run it yourself

1. **Create a Supabase project**, open **SQL Editor**, and run all of `schema.sql`.
2. **Create a `.env` file** with:

   ```env
   SUPABASE_URL=https://<your-project>.supabase.co
   SUPABASE_SERVICE_ROLE_KEY=<service_role key>
   NEXUS_SECRET_KEY=<run: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())">
   GROQ_API_KEY=<your Groq key>        # or GEMINI_API_KEY
   NEXUS_LOGIN_EMAIL=<owner email>
   NEXUS_LOGIN_PASSWORD=<long password>
   APP_TIMEZONE=Asia/Kolkata
   ```

3. **Install and start:**

   ```bash
   pip install -r requirements.txt
   uvicorn main:app --reload --port 8000
   ```

4. Open http://localhost:8000. The first start creates sample products automatically.

To deploy, follow [DEPLOY.md](DEPLOY.md) (Supabase + GitHub + Railway). The start command is in the `Procfile`.

### Optional settings

| Variable | Purpose |
|---|---|
| `ORDER_SIM_SPEED` | Demo clock speed (default 60, use 1 for real time) |
| `LIVE_PLATFORM_APIS` | `1` turns on real platform connection tests |
| `SEED_SAMPLE_DATA` | `0` starts with no sample products |
| `DEMO_LOGIN` | `0` turns off the demo account |
| `AI_REFRESH_HOURS` | Auto-refresh the AI insights after this many hours |
| `MAX_VIDEO_MB` | Largest ad video allowed (default 50) |

---

## 7. Team TechieMechie

- Vatsavaya Suhas Jagapathi Varma
- Neha R Krishna
- PG Pramukh
- Parath S

*Built for DataQuest 3.0.*
