"""Email/password authentication gate backed by Supabase Auth."""
from __future__ import annotations

import requests
from shiny import reactive, render, ui

SUPABASE_URL = "https://vmivonjwubhbvlufxkdd.supabase.co"
SUPABASE_PUBLISHABLE_KEY = "sb_publishable_a3L2nI42Sysrl4GwMfhnRg_pMjnBeEE"


def install_auth(input, output, session, dashboard_ui, dashboard_server):
    user = reactive.Value(None)
    message = reactive.Value("")

    def auth_request(path: str, payload: dict):
        response = requests.post(
            f"{SUPABASE_URL}/auth/v1/{path}",
            headers={
                "apikey": SUPABASE_PUBLISHABLE_KEY,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=15,
        )
        try:
            body = response.json()
        except ValueError:
            body = {}
        if not response.ok:
            detail = body.get("msg") or body.get("message") or body.get("error_description") or "Request failed."
            raise ValueError(str(detail))
        return body

    @render.ui
    def auth_page():
        current = user.get()
        if current:
            return ui.div(
                ui.div(
                    ui.span(f"Signed in as {current['email']}"),
                    ui.input_action_button("auth_logout", "Log out", class_="btn-outline-secondary btn-sm"),
                    class_="auth-toolbar",
                ),
                dashboard_ui,
            )
        return ui.page_fluid(
            ui.div(
                ui.h2("Sign in or create an account"),
                ui.p("Register with your email and verify it before signing in."),
                ui.input_text("auth_email", "Email", placeholder="name@example.com"),
                ui.input_password("auth_password", "Password"),
                ui.div(
                    ui.input_action_button("auth_login", "Sign in", class_="btn-primary"),
                    ui.input_action_button("auth_register", "Register", class_="btn-outline-primary"),
                    class_="auth-actions",
                ),
                ui.output_ui("auth_message"),
                class_="auth-card",
            ),
        )

    @render.ui
    def auth_message():
        text = message.get()
        return ui.p(text, class_="auth-message") if text else None

    @reactive.effect
    @reactive.event(input.auth_register)
    def _register():
        email = (input.auth_email() or "").strip()
        password = input.auth_password() or ""
        if not email or not password:
            message.set("Enter your email and password to register.")
            return
        try:
            result = auth_request("signup", {"email": email, "password": password})
            account = result.get("user") or {}
            if account.get("email_confirmed_at") or result.get("session"):
                message.set("This project may allow unverified sessions. Turn on email confirmation in Supabase Auth settings before using registration.")
                return
            message.set("Registration received. Check your email and confirm your address, then return here to sign in.")
        except (requests.RequestException, ValueError) as exc:
            message.set(str(exc))

    @reactive.effect
    @reactive.event(input.auth_login)
    def _login():
        email = (input.auth_email() or "").strip()
        password = input.auth_password() or ""
        if not email or not password:
            message.set("Enter your email and password to sign in.")
            return
        try:
            result = auth_request("token?grant_type=password", {"email": email, "password": password})
            account = result.get("user") or {}
            if not account.get("email_confirmed_at"):
                message.set("Confirm your email address before signing in.")
                return
            user.set({"email": account.get("email") or email, "access_token": result.get("access_token")})
            message.set("")
        except (requests.RequestException, ValueError) as exc:
            message.set(str(exc))

    @reactive.effect
    @reactive.event(input.auth_logout)
    def _logout():
        current = user.get()
        if current and current.get("access_token"):
            try:
                requests.post(
                    f"{SUPABASE_URL}/auth/v1/logout",
                    headers={
                        "apikey": SUPABASE_PUBLISHABLE_KEY,
                        "Authorization": f"Bearer {current['access_token']}",
                    },
                    timeout=10,
                )
            except requests.RequestException:
                pass
        user.set(None)
        message.set("You have signed out.")

    dashboard_server(input, output, session)
