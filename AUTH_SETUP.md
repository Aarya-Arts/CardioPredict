# CardioPredict email authentication setup

CardioPredict uses Supabase Auth for email/password registration and sign-in.

## 1. Create a Supabase project
1. Open https://supabase.com/dashboard and create a project.
2. In Project Settings → API, copy the Project URL and the public anon/publishable key.
3. In Authentication → Providers, make sure the Email provider is enabled.
4. Choose whether email confirmation is required. If enabled, users must click the confirmation email before signing in.
5. In Authentication → URL Configuration, set the Site URL to your deployed Streamlit app URL and add that URL to the redirect allow list.

## 2. Add Streamlit secrets
In Streamlit Community Cloud, open the app's Settings → Secrets and add:

```toml
SUPABASE_URL = "https://YOUR-PROJECT-REF.supabase.co"
SUPABASE_ANON_KEY = "YOUR_PUBLIC_ANON_KEY"
```

For local development, create `.streamlit/secrets.toml` with the same two entries. That file is ignored by Git.

**Never add a Supabase service-role/secret key to this app, GitHub, or a committed file.** The public anon key is intended for client applications; access is governed by Supabase Auth and database policies.

## 3. Install and run
```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

## 4. Test
- Register with an email address you control.
- If email confirmation is enabled, confirm the address from your inbox.
- Sign in with the same email and password.
- Confirm the dashboard loads and the Sign out button returns to the login screen.

## Notes
- User accounts and password verification are managed by Supabase; CardioPredict does not store plain-text passwords.
- The app keeps the authenticated session in Streamlit's server-side session state. A new browser session may require signing in again.
- Supabase's default email sender has rate limits. Configure custom SMTP in Supabase for reliable production email delivery.
- The existing ML model and Gemini assistant remain separate from authentication.
