"""Supabase email authentication for CardioPredict.

Only the Supabase project URL and public anon/publishable key belong in
Streamlit Secrets. Never use a service-role key in this app.
"""
import streamlit as st
from supabase import create_client


def _credentials():
    try:
        url = str(st.secrets.get("SUPABASE_URL", "")).strip()
        key = str(st.secrets.get("SUPABASE_ANON_KEY", "")).strip()
    except Exception:
        url, key = "", ""
    return url, key


def _client():
    url, key = _credentials()
    if not url or not key:
        return None
    client = create_client(url, key)
    access = st.session_state.get("cp_supabase_access")
    refresh = st.session_state.get("cp_supabase_refresh")
    if access and refresh:
        try:
            session = client.auth.set_session(access, refresh)
            if getattr(session, "session", None):
                st.session_state["cp_supabase_access"] = session.session.access_token
                st.session_state["cp_supabase_refresh"] = session.session.refresh_token
        except Exception:
            _clear_session()
    return client


def _clear_session():
    for name in (
        "cp_supabase_access",
        "cp_supabase_refresh",
        "cp_auth_email",
        "cp_auth_name",
    ):
        st.session_state.pop(name, None)


def _save_session(response):
    session = getattr(response, "session", None)
    user = getattr(response, "user", None)
    if session is None or user is None:
        return False
    st.session_state["cp_supabase_access"] = session.access_token
    st.session_state["cp_supabase_refresh"] = session.refresh_token
    st.session_state["cp_auth_email"] = getattr(user, "email", "") or ""
    metadata = getattr(user, "user_metadata", {}) or {}
    st.session_state["cp_auth_name"] = metadata.get("full_name", "")
    return True


def _auth_error(exc):
    message = str(exc)
    safe_messages = (
        ("Invalid login credentials", "Email or password is incorrect."),
        ("Email not confirmed", "Please confirm your email using the link we sent you."),
        ("User already registered", "An account with this email already exists. Please sign in."),
        ("Password should be at least", "Choose a stronger password."),
        ("Unable to validate email", "Please enter a valid email address."),
    )
    for needle, friendly in safe_messages:
        if needle.lower() in message.lower():
            return friendly
    return "Authentication could not be completed. Check your details and try again."


def render_auth_gate():
    """Render the branded login/registration screen; return True when signed in."""
    if st.session_state.get("cp_supabase_access") and st.session_state.get("cp_auth_email"):
        # Revalidate/refresh the user's session on each app run.
        client = _client()
        if client is not None:
            try:
                client.auth.get_user()
                return True
            except Exception:
                _clear_session()

    url, key = _credentials()
    st.markdown(
        """
        <div class="auth-shell">
          <section class="auth-brand">
            <div>
              <div class="auth-mark">CARDIO<span>PREDICT</span></div>
              <div class="eyebrow" style="margin-top:2.2rem">HEART &amp; LIFE · HEALTH INTELLIGENCE</div>
              <h1>Understand your heart. Make knowledge matter.</h1>
              <p>A thoughtful space to explore cardiovascular data, learn how predictive models work, and build a clearer understanding of heart-health concepts.</p>
            </div>
            <div class="trust">13 model features &nbsp;·&nbsp; Machine learning &nbsp;·&nbsp; Educational AI</div>
          </section>
          <section class="auth-panel">
            <div class="section-kicker">YOUR PRIVATE WORKSPACE</div>
            <h2>Welcome to CardioPredict</h2>
            <p>Sign in or create an account to continue.</p>
        """,
        unsafe_allow_html=True,
    )

    if not url or not key:
        st.warning(
            "Email authentication is not connected yet. Add SUPABASE_URL and "
            "SUPABASE_ANON_KEY to Streamlit Secrets to activate account sign-in."
        )
        st.markdown(
            "Open the [Supabase dashboard](https://supabase.com/dashboard), create "
            "a project, then copy its Project URL and anon/public key into "
            "Streamlit Cloud → App settings → Secrets. Never use the service-role key."
        )
        st.markdown("</section></div>", unsafe_allow_html=True)
        return False

    client = _client()
    if client is None:
        st.error("Authentication could not be initialized. Check the Supabase settings.")
        st.markdown("</section></div>", unsafe_allow_html=True)
        return False

    sign_in, sign_up, reset = st.tabs(["Sign in", "Create account", "Reset password"])
    with sign_in:
        with st.form("cp_sign_in_form", clear_on_submit=False):
            email = st.text_input("Email address", key="cp_login_email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", key="cp_login_password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign in securely", use_container_width=True)
        if submitted:
            if not email.strip() or not password:
                st.error("Enter both your email and password.")
            else:
                try:
                    response = client.auth.sign_in_with_password(
                        {"email": email.strip(), "password": password}
                    )
                    if _save_session(response):
                        st.success("Signed in successfully.")
                        st.rerun()
                    st.error("Sign-in did not return a session. Check your email confirmation settings.")
                except Exception as exc:
                    st.error(_auth_error(exc))

    with sign_up:
        with st.form("cp_sign_up_form", clear_on_submit=False):
            new_email = st.text_input("Email address", key="cp_register_email", placeholder="you@example.com")
            new_password = st.text_input("Create password", type="password", key="cp_register_password", placeholder="At least 8 characters")
            confirm = st.text_input("Confirm password", type="password", key="cp_register_confirm", placeholder="Re-enter your password")
            registered = st.form_submit_button("Create my account", use_container_width=True)
        if registered:
            if not new_email.strip() or not new_password or not confirm:
                st.error("Complete all three fields.")
            elif len(new_password) < 8:
                st.error("Use a password with at least 8 characters.")
            elif new_password != confirm:
                st.error("The passwords do not match.")
            else:
                try:
                    response = client.auth.sign_up(
                        {"email": new_email.strip(), "password": new_password}
                    )
                    if _save_session(response):
                        st.success("Your account is ready. Welcome to CardioPredict!")
                        st.rerun()
                    else:
                        st.success("Account created. Check your inbox for the confirmation link, then sign in.")
                except Exception as exc:
                    st.error(_auth_error(exc))

    with reset:
        st.caption("We'll send a password-reset link if the email belongs to an account.")
        with st.form("cp_reset_form", clear_on_submit=True):
            reset_email = st.text_input("Account email", key="cp_reset_email", placeholder="you@example.com")
            reset_submitted = st.form_submit_button("Send reset link", use_container_width=True)
        if reset_submitted:
            if not reset_email.strip():
                st.error("Enter your account email.")
            else:
                try:
                    client.auth.reset_password_for_email(reset_email.strip())
                    st.success("If the account exists, a password-reset email has been requested.")
                except Exception:
                    st.error("The reset email could not be sent. Check your Supabase email settings.")

    st.markdown(
        '<p class="trust" style="margin-top:1.4rem">Your password is handled by Supabase Authentication. '
        'CardioPredict does not store plain-text passwords. This prototype is educational, not a medical service.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("</section></div>", unsafe_allow_html=True)
    return False


def render_logout_control():
    """Show the signed-in account and provide a safe sign-out action."""
    email = st.session_state.get("cp_auth_email", "")
    if email:
        st.caption("Signed in as")
        st.markdown(f"**{email}**")
    if st.button("Sign out", key="cp_sign_out", use_container_width=True):
        client = _client()
        if client is not None:
            try:
                client.auth.sign_out()
            except Exception:
                pass
        _clear_session()
        st.rerun()
