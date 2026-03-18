from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
]


@dataclass(frozen=True)
class UnreadEmail:
    id: str
    thread_id: str
    message_count: int
    subject: str
    sender: str
    body_text: str


def _get_or_create_label_id(service: Any, *, user_id: str, label_name: str) -> str:
    name = (label_name or "").strip()
    if not name:
        raise ValueError("label_name must be non-empty")

    resp = service.users().labels().list(userId=user_id).execute()
    for lab in resp.get("labels") or []:
        if (lab.get("name") or "").strip().lower() == name.lower():
            return str(lab.get("id"))

    created = (
        service.users()
        .labels()
        .create(
            userId=user_id,
            body={
                "name": name,
                "labelListVisibility": "labelShow",
                "messageListVisibility": "show",
            },
        )
        .execute()
    )
    return str(created.get("id"))


def mark_as_read(service: Any, email_id: str, *, user_id: str = "me") -> Dict[str, Any]:
    """
    Marks an email as read by removing the UNREAD label.
    """
    return (
        service.users()
        .messages()
        .modify(userId=user_id, id=email_id, body={"addLabelIds": [], "removeLabelIds": ["UNREAD"]})
        .execute()
    )


def mark_as_bot_processed(service: Any, email_id: str, *, user_id: str = "me") -> Dict[str, Any]:
    """
    Adds a Gmail label called 'Bot-Processed' to an email.
    If the label doesn't exist, creates it first.
    """
    label_id = _get_or_create_label_id(service, user_id=user_id, label_name="Bot-Processed")
    return (
        service.users()
        .messages()
        .modify(userId=user_id, id=email_id, body={"addLabelIds": [label_id], "removeLabelIds": []})
        .execute()
    )


def mark_needs_human(service: Any, thread_id: str, *, user_id: str = "me") -> Dict[str, Any]:
    """
    Adds a Gmail label called 'Needs-Human' to a thread.
    If the label doesn't exist, creates it first.
    """
    label_id = _get_or_create_label_id(service, user_id=user_id, label_name="Needs-Human")
    return (
        service.users()
        .threads()
        .modify(userId=user_id, id=thread_id, body={"addLabelIds": [label_id], "removeLabelIds": []})
        .execute()
    )


def _b64url_decode(data: str) -> bytes:
    # Gmail uses URL-safe base64 without padding.
    missing = (-len(data)) % 4
    if missing:
        data += "=" * missing
    return base64.urlsafe_b64decode(data.encode("utf-8"))


def _extract_header(headers: List[Dict[str, str]], name: str) -> str:
    name_l = name.lower()
    for h in headers:
        if h.get("name", "").lower() == name_l:
            return h.get("value", "") or ""
    return ""


def _extract_text_plain_from_payload(payload: Dict[str, Any]) -> str:
    """
    Attempts to extract a best-effort text body from a Gmail message payload.
    Prefers text/plain; falls back to stripping nothing from text/html if needed.
    """
    if not payload:
        return ""

    mime_type = payload.get("mimeType", "")
    body = payload.get("body") or {}
    data = body.get("data")
    if data and mime_type.startswith("text/"):
        try:
            return _b64url_decode(data).decode("utf-8", errors="replace").strip()
        except Exception:
            return ""

    parts = payload.get("parts") or []
    if not parts:
        return ""

    # First pass: find text/plain
    for part in parts:
        part_mime = part.get("mimeType", "")
        if part_mime == "text/plain":
            pbody = part.get("body") or {}
            pdata = pbody.get("data")
            if pdata:
                try:
                    return _b64url_decode(pdata).decode("utf-8", errors="replace").strip()
                except Exception:
                    return ""

    # Second pass: recurse into nested multiparts
    for part in parts:
        got = _extract_text_plain_from_payload(part)
        if got:
            return got

    # Last resort: text/html
    for part in parts:
        if part.get("mimeType") == "text/html":
            pbody = part.get("body") or {}
            pdata = pbody.get("data")
            if pdata:
                try:
                    return _b64url_decode(pdata).decode("utf-8", errors="replace").strip()
                except Exception:
                    return ""

    return ""


class GmailClient:
    def __init__(
        self,
        *,
        credentials_file: str | Path = "credentials.json",
        token_file: str | Path = "token.json",
        user_id: str = "me",
        scopes: Optional[List[str]] = None,
    ) -> None:
        load_dotenv()
        self.user_id = user_id
        self.credentials_file = Path(os.getenv("GOOGLE_CREDENTIALS_FILE", str(credentials_file)))
        self.token_file = Path(token_file)
        self.scopes = scopes or SCOPES

        self._service = None

    def authenticate(self) -> Any:
        """
        Authenticates with Gmail API using OAuth client secrets in credentials.json.
        Saves/refreshes token.json for future use.
        Returns the gmail service client.
        """
        creds: Optional[Credentials] = None
        if self.token_file.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_file), self.scopes)

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        elif not creds or not creds.valid:
            if not self.credentials_file.exists():
                raise FileNotFoundError(f"Missing OAuth client file: {self.credentials_file}")
            flow = InstalledAppFlow.from_client_secrets_file(str(self.credentials_file), self.scopes)
            creds = flow.run_local_server(port=0)

        self.token_file.write_text(creds.to_json(), encoding="utf-8")
        self._service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        return self._service

    @property
    def service(self) -> Any:
        if self._service is None:
            return self.authenticate()
        return self._service

    def poll_unread_inbox(self, *, max_results: int = 10) -> List[UnreadEmail]:
        """
        Polls the inbox for unread emails (does not mark them read).
        Returns list of UnreadEmail with id (latest message id), thread_id, message_count,
        subject/sender/body text extracted from the latest message in the thread.
        """
        resp = (
            self.service.users()
            .threads()
            .list(
                userId=self.user_id,
                labelIds=["INBOX"],
                q='is:unread -label:"Bot-Processed"',
                maxResults=max_results,
            )
            .execute()
        )

        threads = resp.get("threads") or []
        out: List[UnreadEmail] = []

        for t in threads:
            thread_id = t.get("id")
            if not thread_id:
                continue

            full = (
                self.service.users()
                .threads()
                .get(userId=self.user_id, id=thread_id, format="full")
                .execute()
            )

            thread_messages = full.get("messages") or []
            message_count = len(thread_messages)
            if message_count == 0:
                continue

            # Choose the latest message in the thread by internalDate.
            def _internal_date(m: Dict[str, Any]) -> int:
                try:
                    return int(m.get("internalDate") or 0)
                except Exception:
                    return 0

            latest = max(thread_messages, key=_internal_date)
            latest_id = str(latest.get("id", "")) or ""
            payload = latest.get("payload") or {}
            headers = payload.get("headers") or []
            subject = _extract_header(headers, "Subject")
            sender = _extract_header(headers, "From")
            body_text = _extract_text_plain_from_payload(payload)

            out.append(
                UnreadEmail(
                    id=latest_id,
                    thread_id=str(full.get("id", thread_id)),
                    message_count=message_count,
                    subject=subject,
                    sender=sender,
                    body_text=body_text,
                )
            )

        return out

    def add_label(self, email_id: str, label_id: str) -> Dict[str, Any]:
        """
        Adds an existing label (by labelId) to an email.
        """
        return (
            self.service.users()
            .messages()
            .modify(userId=self.user_id, id=email_id, body={"addLabelIds": [label_id], "removeLabelIds": []})
            .execute()
        )

    def get_or_create_label_id(self, label_name: str) -> str:
        """
        Returns the Gmail labelId for the given label name.
        Creates the label if it does not exist.
        """
        name = (label_name or "").strip()
        if not name:
            raise ValueError("label_name must be non-empty")

        resp = self.service.users().labels().list(userId=self.user_id).execute()
        for lab in resp.get("labels") or []:
            if (lab.get("name") or "").strip().lower() == name.lower():
                return str(lab.get("id"))

        created = (
            self.service.users()
            .labels()
            .create(
                userId=self.user_id,
                body={
                    "name": name,
                    "labelListVisibility": "labelShow",
                    "messageListVisibility": "show",
                },
            )
            .execute()
        )
        return str(created.get("id"))

    def mark_as_read(self, email_id: str) -> Dict[str, Any]:
        """
        Removes the UNREAD label from a message.
        """
        return (
            self.service.users()
            .messages()
            .modify(userId=self.user_id, id=email_id, body={"addLabelIds": [], "removeLabelIds": ["UNREAD"]})
            .execute()
        )

    @staticmethod
    def extract_email_address(from_header: str) -> str:
        """
        Extracts the email address from a From: header (best-effort).
        Example: 'Jane Doe <jane@example.com>' -> 'jane@example.com'
        """
        s = (from_header or "").strip()
        m = re.search(r"<([^>]+)>", s)
        if m:
            return m.group(1).strip()
        return s

    def reply_to_thread(
        self,
        *,
        email_id: str,
        thread_id: str,
        to: str,
        subject: str,
        body_text: str,
        cc: Optional[str] = None,
        bcc: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Replies to an email thread.

        You must pass:
        - email_id: the message id you are replying to (used to fetch Message-Id)
        - thread_id: the Gmail thread id
        - to: recipient address (usually original sender)
        - subject: original subject; "Re:" is added if missing
        - body_text: reply body (plain text)
        """
        original = (
            self.service.users()
            .messages()
            .get(userId=self.user_id, id=email_id, format="metadata", metadataHeaders=["Message-Id", "References"])
            .execute()
        )
        opayload = original.get("payload") or {}
        oheaders = opayload.get("headers") or []
        message_id_header = _extract_header(oheaders, "Message-Id")
        references_header = _extract_header(oheaders, "References")

        reply_subject = subject.strip()
        if reply_subject and not reply_subject.lower().startswith("re:"):
            reply_subject = f"Re: {reply_subject}"

        msg = EmailMessage()
        msg["To"] = to
        msg["Subject"] = reply_subject or "Re:"
        if cc:
            msg["Cc"] = cc
        if bcc:
            msg["Bcc"] = bcc
        if message_id_header:
            msg["In-Reply-To"] = message_id_header
            # Append original Message-Id to References if present.
            if references_header:
                msg["References"] = f"{references_header} {message_id_header}".strip()
            else:
                msg["References"] = message_id_header
        msg.set_content(body_text or "")

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
        body = {"raw": raw, "threadId": thread_id}

        return self.service.users().messages().send(userId=self.user_id, body=body).execute()

