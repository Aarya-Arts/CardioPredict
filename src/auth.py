"""Supabase email authentication for CardioPredict.

Only the Supabase project URL and public anon/publishable key belong in
Streamlit Secrets. Never use a service-role key in this app.
"""
from pathlib import Path

import streamlit as st
from supabase import create_client


def _brand_svg():
    """Read the shared CardioPredict brand emblem for the sign-in screen."""
    asset = Path(__file__).resolve().parents[1] / "assets" / "cardiopredict-mark.svg"
    try:
        return asset.read_text(encoding="utf-8")
    except OSError:
        return ""


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
        client = _client()
        if client is not None:
            try:
                client.auth.get_user()
                return True
            except Exception:
                _clear_session()

    url, key = _credentials()
    brand_svg = _brand_svg()
    left, right = st.columns([1.02, 0.98], gap="large")
    with left:
        st.markdown(
            f"""
            <div class="auth-brand">
              <div>
                <div class="auth-brand-lockup">{brand_svg}<div><div class="auth-mark">Cardio<span>Predict</span></div><div class="auth-brand-subtitle">H E A R T &nbsp; &amp; &nbsp; L I F E</div></div></div>
                <div class="eyebrow" style="margin-top:2.2rem">HEART &amp; LIFE · HEALTH INTELLIGENCE</div>
                <h1>Understand your heart.<br>Make knowledge matter.</h1>
                <p>A thoughtful space to explore cardiovascular data, learn how predictive models work, and build a clearer understanding of heart-health concepts.</p>
              </div>
              <div class="trust">12 model features &nbsp;·&nbsp; Machine learning &nbsp;·&nbsp; Educational AI</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="section-kicker">YOUR PRIVATE WORKSPACE</div>', unsafe_allow_html=True)
        st.markdown("## Welcome to CardioPredict")
        st.caption("Sign in or create an account to continue.")

        if not url or not key:
            st.warning(
                "Email authentication is not connected yet. Add SUPABASE_URL and "
                "SUPABASE_ANON_KEY to Streamlit Secrets to activate account sign-in."
            )
            st.markdown(
                "Open the [Supabase dashboard](https://supabase.com/dashboard), create "
                "a project, then add its Project URL and anon/public key to "
                "Streamlit Cloud → App settings → Secrets. Never use the service-role key."
            )
            return False

        client = _client()
        if client is None:
            st.error("Authentication could not be initialized. Check the Supabase settings.")
            return False

        sign_in, sign_up = st.tabs(["Sign in", "Create account"])
        with sign_in:
            remember_email = st.checkbox(
                "Remember my email during this app session (never save my password)",
                key="cp_remember_email_choice",
            )
            saved_email = st.session_state.get("cp_saved_email", "")
            if remember_email and saved_email and not st.session_state.get("cp_login_email"):
                st.session_state["cp_login_email"] = saved_email
            with st.form("cp_sign_in_form", clear_on_submit=False):
                email = st.text_input(
                    "Email address",
                    key="cp_login_email",
                    placeholder="you@example.com",
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    key="cp_login_password",
                    placeholder="Enter your password",
                )
                submitted = st.form_submit_button(
                    "Sign in securely", use_container_width=True
                )
            if submitted:
                if not email.strip() or not password:
                    st.error("Enter both your email and password.")
                else:
                    try:
                        response = client.auth.sign_in_with_password(
                            {"email": email.strip(), "password": password}
                        )
                        if _save_session(response):
                            if remember_email:
                                st.session_state["cp_saved_email"] = email.strip()
                            else:
                                st.session_state.pop("cp_saved_email", None)
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

        st.markdown(
            '<p class="trust" style="margin-top:1.2rem">Passwords are managed by Supabase Authentication. '
            'CardioPredict does not store plain-text passwords. Educational prototype, not a medical service.</p>',
            unsafe_allow_html=True,
        )
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
