from __future__ import annotations

import os
import time
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

from agent import generate_reply
from classifier import classify_email
from gmail_client import GmailClient, mark_as_bot_processed, mark_as_read, mark_needs_human


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _get_poll_interval_minutes() -> int:
    load_dotenv()
    raw = os.getenv("EMAIL_POLL_INTERVAL_MINUTES", "2").strip()
    try:
        minutes = int(raw)
        return max(1, minutes)
    except Exception:
        return 2


def poll_and_process() -> None:
    _log("Polling Gmail for unread emails...")

    gmail = GmailClient()
    gmail.authenticate()

    auto_replied_label_id = gmail.get_or_create_label_id("Auto-Replied")
    unread = gmail.poll_unread_inbox(max_results=20)
    _log(f"Found {len(unread)} unread email(s).")

    for email in unread:
        _log(
            f"Processing email_id={email.id} thread_id={email.thread_id} "
            f"message_count={email.message_count} subject={email.subject!r} from={email.sender!r}"
        )

        if email.message_count > 1:
            _log("Thread has multiple messages. Escalating to human and bot-marking (no classification/reply).")
            mark_needs_human(gmail.service, email.thread_id, user_id=gmail.user_id)
            mark_as_bot_processed(gmail.service, email.id, user_id=gmail.user_id)
            continue

        label = classify_email(email.body_text)
        _log(f"Classified as: {label}")

        if label != "Customer Support":
            _log("Classified as Other. Marking as Bot-Processed (leaving unread).")
            mark_as_bot_processed(gmail.service, email.id, user_id=gmail.user_id)
            continue

        _log("Drafting reply with support agent (may use FAQ/policy lookup)...")
        reply_body = generate_reply(email.body_text)

        if isinstance(reply_body, dict) and reply_body.get("escalate") is True:
            _log("No relevant KB answer found. Escalating to human (Needs-Human).")
            mark_needs_human(gmail.service, email.thread_id, user_id=gmail.user_id)
            continue

        to_addr = GmailClient.extract_email_address(email.sender)
        _log(f"Sending reply to thread_id={email.thread_id} to={to_addr!r}")
        gmail.reply_to_thread(
            email_id=email.id,
            thread_id=email.thread_id,
            to=to_addr,
            subject=email.subject,
            body_text=reply_body,
        )

        _log("Adding label 'Auto-Replied' to email...")
        gmail.add_label(email.id, auto_replied_label_id)

        _log("Marking email as read...")
        mark_as_read(gmail.service, email.id, user_id=gmail.user_id)

        _log("Marking email as Bot-Processed...")
        mark_as_bot_processed(gmail.service, email.id, user_id=gmail.user_id)

        _log("Done.")


def main() -> None:
    interval_minutes = _get_poll_interval_minutes()
    _log(f"Starting scheduler. Poll interval: {interval_minutes} minute(s).")

    scheduler = BackgroundScheduler()
    scheduler.add_job(poll_and_process, "interval", minutes=interval_minutes, max_instances=1, coalesce=True)
    scheduler.start()

    _log("Scheduler started. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        _log("Stopping scheduler...")
        scheduler.shutdown(wait=False)
        _log("Stopped.")


if __name__ == "__main__":
    main()

