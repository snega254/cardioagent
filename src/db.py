"""
MongoDB Atlas database layer for CardioAgent's application data (users,
ECG records, analyses, reports). Deliberately separate from the medical
knowledge base (FAISS/vector_store.pkl) — no patient data is ever written
there, and no medical-knowledge text is ever written here.

CONNECTION SETUP (you must do this — I cannot create an Atlas account or
test real connectivity from my sandbox, which has no network route to
MongoDB Atlas):

1. Create a free MongoDB Atlas account at https://www.mongodb.com/cloud/atlas
2. Create a free (M0) cluster.
3. Database Access -> add a database user with a password.
4. Network Access -> add your current IP address (or 0.0.0.0/0 for local
   dev only — not recommended beyond testing).
5. Connect -> "Connect your application" -> copy the connection string,
   which looks like:
   mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/
6. Put it in `.streamlit/secrets.toml` (create this file, do NOT commit
   it to git — it's already in .gitignore):
   MONGODB_URI = "mongodb+srv://..."

This module reads that URI via st.secrets, with an environment-variable
fallback for non-Streamlit usage (e.g. running tests directly).
"""
import os
from datetime import datetime, timezone

from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

DB_NAME = "cardioagent"


def get_mongo_uri():
    try:
        import streamlit as st
        if "MONGODB_URI" in st.secrets:
            return st.secrets["MONGODB_URI"]
    except Exception:
        pass
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise RuntimeError(
            "No MongoDB connection string found. Set MONGODB_URI in "
            ".streamlit/secrets.toml (for the app) or as an environment "
            "variable (for scripts/tests). See the top of db.py for setup steps."
        )
    return uri


class CardioDB:
    """Wraps a pymongo client. Pass a client directly (e.g. mongomock) for
    testing without a real Atlas connection."""

    def __init__(self, client=None, uri=None):
        if client is not None:
            self.client = client
        else:
            uri = uri or get_mongo_uri()
            self.client = MongoClient(uri, serverSelectionTimeoutMS=8000)
            try:
                self.client.admin.command("ping")
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                raise RuntimeError(
                    f"Could not connect to MongoDB Atlas: {e}\n"
                    f"Check: (1) your connection string in "
                    f".streamlit/secrets.toml, (2) that your current IP is "
                    f"whitelisted in Atlas Network Access, (3) your internet "
                    f"connection."
                )
        self.db = self.client[DB_NAME]
        self.users = self.db["users"]
        self.ecg_records = self.db["ecg_records"]
        self.analyses = self.db["analyses"]
        self.reports = self.db["reports"]
        self._ensure_indexes()

    def _ensure_indexes(self):
        self.users.create_index("username", unique=True)

    def now(self):
        return datetime.now(timezone.utc).isoformat()

    # ---------------- Users ----------------

    def create_user(self, username, password_hash, name, age, sex):
        doc = {
            "username": username, "password_hash": password_hash,
            "name": name, "age": age, "sex": sex, "created_at": self.now(),
        }
        result = self.users.insert_one(doc)
        return str(result.inserted_id)

    def get_user_by_username(self, username):
        doc = self.users.find_one({"username": username})
        if doc is None:
            return None
        doc["id"] = str(doc.pop("_id"))
        return doc

    # ---------------- ECG records ----------------

    def create_ecg_record(self, user_id, filename, source_type, sampling_rate,
                           n_leads, duration_sec, file_dir=None):
        doc = {
            "user_id": user_id, "filename": filename, "source_type": source_type,
            "sampling_rate": sampling_rate, "n_leads": n_leads,
            "duration_sec": duration_sec, "upload_time": self.now(),
            "file_dir": file_dir,
        }
        result = self.ecg_records.insert_one(doc)
        return str(result.inserted_id)

    def get_ecg_record(self, ecg_record_id):
        doc = self.ecg_records.find_one({"_id": ObjectId(ecg_record_id)})
        if doc is None:
            return None
        doc["id"] = str(doc.pop("_id"))
        return doc

    def get_ecg_records_for_user(self, user_id):
        docs = list(self.ecg_records.find({"user_id": user_id}).sort("upload_time", -1))
        for d in docs:
            d["id"] = str(d.pop("_id"))
        return docs

    # ---------------- Analyses ----------------

    def create_analysis(self, ecg_record_id, prediction, confidence, features,
                         xai, rag_sources, explanation):
        doc = {
            "ecg_record_id": ecg_record_id, "prediction": prediction,
            "confidence": confidence, "features": features, "xai": xai,
            "rag_sources": rag_sources, "explanation": explanation,
            "created_at": self.now(),
        }
        result = self.analyses.insert_one(doc)
        return str(result.inserted_id)

    def get_analysis(self, analysis_id):
        doc = self.analyses.find_one({"_id": ObjectId(analysis_id)})
        if doc is None:
            return None
        doc["id"] = str(doc.pop("_id"))
        return doc

    # ---------------- Reports ----------------

    def create_report(self, user_id, ecg_record_id, analysis_id, report_metadata,
                       report_file_path):
        doc = {
            "user_id": user_id, "ecg_record_id": ecg_record_id,
            "analysis_id": analysis_id, "report_metadata": report_metadata,
            "report_file_path": report_file_path, "created_at": self.now(),
        }
        result = self.reports.insert_one(doc)
        return str(result.inserted_id)

    def get_reports_for_user(self, user_id):
        """Returns reports with their ecg_record and analysis data attached
        (manual lookups instead of a native JOIN — fine at this scale)."""
        reports = list(self.reports.find({"user_id": user_id}).sort("created_at", -1))
        results = []
        for r in reports:
            r["id"] = str(r.pop("_id"))
            r["ecg_record"] = self.get_ecg_record(r["ecg_record_id"])
            r["analysis"] = self.get_analysis(r["analysis_id"])
            results.append(r)
        return results

    def get_report_by_id(self, report_id):
        r = self.reports.find_one({"_id": ObjectId(report_id)})
        if r is None:
            return None
        r["id"] = str(r.pop("_id"))
        r["ecg_record"] = self.get_ecg_record(r["ecg_record_id"])
        r["analysis"] = self.get_analysis(r["analysis_id"])
        return r
