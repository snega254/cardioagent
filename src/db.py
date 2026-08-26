"""
MongoDB Atlas database layer for CardioAgent's application data.
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
            ".streamlit/secrets.toml or as an environment variable."
        )
    return uri


class CardioDB:
    def __init__(self, client=None, uri=None):
        if client is not None:
            self.client = client
        else:
            uri = uri or get_mongo_uri()
            self.client = MongoClient(uri, serverSelectionTimeoutMS=8000)
            try:
                self.client.admin.command("ping")
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                raise RuntimeError(f"Could not connect to MongoDB Atlas: {e}")
        self.db = self.client[DB_NAME]
        self.users = self.db["users"]
        self.ecg_records = self.db["ecg_records"]
        self.analyses = self.db["analyses"]
        self.reports = self.db["reports"]
        self.patients = self.db["patients"]  # Add patients collection
        self._ensure_indexes()

    def _ensure_indexes(self):
        # Drop old username index if it exists (from previous version)
        try:
            self.users.drop_index("username_1")
        except:
            pass  # Index doesn't exist
        
        # Create email unique index
        self.users.create_index("email", unique=True)
        
        # Create patient indexes
        self.patients.create_index("user_id")
        self.patients.create_index("name")

    def now(self):
        return datetime.now(timezone.utc).isoformat()

    # ============================================================
    # USERS (Email-based)
    # ============================================================

    def create_user(self, email, password_hash, name):
        # Check if user already exists
        existing = self.users.find_one({"email": email})
        if existing:
            raise ValueError(f"User with email {email} already exists")
        
        doc = {
            "email": email,
            "password_hash": password_hash,
            "name": name,
            "created_at": self.now(),
        }
        result = self.users.insert_one(doc)
        return str(result.inserted_id)

    def get_user_by_email(self, email):
        doc = self.users.find_one({"email": email})
        if doc is None:
            return None
        doc["id"] = str(doc.pop("_id"))
        return doc

    # ============================================================
    # PATIENTS
    # ============================================================

    def create_patient(self, user_id, name, age, sex, symptoms):
        """Create a new patient."""
        doc = {
            "user_id": user_id,
            "name": name.strip(),
            "age": age,
            "sex": sex,
            "symptoms": symptoms or [],
            "symptoms_text": ", ".join(symptoms) if symptoms else "",
            "created_at": self.now(),
            "updated_at": self.now()
        }
        result = self.patients.insert_one(doc)
        return str(result.inserted_id)

    def get_patient(self, patient_id):
        """Get a patient by ID."""
        doc = self.patients.find_one({"_id": ObjectId(patient_id)})
        if doc:
            doc["id"] = str(doc.pop("_id"))
        return doc

    def get_patients_for_user(self, user_id):
        """Get all patients for a user."""
        docs = list(self.patients.find({"user_id": user_id}).sort("created_at", -1))
        for doc in docs:
            doc["id"] = str(doc.pop("_id"))
        return docs

    def update_patient(self, patient_id, **kwargs):
        """Update patient information."""
        update_data = {k: v for k, v in kwargs.items() if v is not None}
        if not update_data:
            return False
        update_data["updated_at"] = self.now()
        if "symptoms" in update_data and isinstance(update_data["symptoms"], list):
            update_data["symptoms_text"] = ", ".join(update_data["symptoms"])
        result = self.patients.update_one(
            {"_id": ObjectId(patient_id)},
            {"$set": update_data}
        )
        return result.modified_count > 0

    def delete_patient(self, patient_id):
        """Delete a patient and associated analyses."""
        # Delete associated analyses
        self.analyses.delete_many({"patient_id": patient_id})
        # Delete the patient
        result = self.patients.delete_one({"_id": ObjectId(patient_id)})
        return result.deleted_count > 0

    def get_analyses_for_patient(self, patient_id):
        """Get all analyses for a patient."""
        docs = list(self.analyses.find({"patient_id": patient_id}).sort("created_at", -1))
        for doc in docs:
            doc["id"] = str(doc.pop("_id"))
        return docs

    # ============================================================
    # ECG RECORDS
    # ============================================================

    def create_ecg_record(self, user_id, filename, source_type, sampling_rate,
                           n_leads, duration_sec, file_dir=None, patient_info=None):
        doc = {
            "user_id": user_id,
            "filename": filename,
            "source_type": source_type,
            "sampling_rate": sampling_rate,
            "n_leads": n_leads,
            "duration_sec": duration_sec,
            "upload_time": self.now(),
            "file_dir": file_dir,
            "patient_info": patient_info or {},
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
        for doc in docs:
            doc["id"] = str(doc.pop("_id"))
        return docs

    # ============================================================
    # ANALYSES
    # ============================================================

    def create_analysis(self, ecg_record_id, prediction, confidence, features,
                         xai, rag_sources, explanation, severity, summary,
                         mode_type="research", patient_context=None, 
                         clinical_reasoning=None, report_text=None,
                         medications=None, clinical_considerations=None):
        """Create an analysis linked to an ECG record."""
        doc = {
            "ecg_record_id": ecg_record_id,
            "prediction": prediction,
            "confidence": confidence,
            "features": features,
            "xai": xai,
            "rag_sources": rag_sources,
            "explanation": explanation,
            "severity": severity,
            "summary": summary,
            "mode_type": mode_type,
            "patient_context": patient_context or {},
            "clinical_reasoning": clinical_reasoning or {},
            "report_text": report_text,
            "medications": medications or [],
            "clinical_considerations": clinical_considerations or [],
            "created_at": self.now(),
        }
        result = self.analyses.insert_one(doc)
        return str(result.inserted_id)

    def create_analysis_with_patient(self, patient_id, user_id, analysis_type, 
                                      prediction, confidence, features, xai, 
                                      rag_sources, explanation, severity, summary,
                                      mode_type="signal", report_text=None,
                                      patient_context=None, clinical_reasoning=None):
        """
        Create an analysis linked directly to a patient.
        """
        doc = {
            "patient_id": patient_id,
            "user_id": user_id,
            "analysis_type": analysis_type,  # "signal", "report", "image"
            "prediction": prediction,
            "confidence": confidence,
            "features": features,
            "xai": xai,
            "rag_sources": rag_sources,
            "explanation": explanation,
            "severity": severity,
            "summary": summary,
            "mode_type": mode_type,
            "patient_context": patient_context or {},
            "clinical_reasoning": clinical_reasoning or {},
            "report_text": report_text,
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

    def get_analysis_by_id(self, analysis_id):
        """Get an analysis by ID with patient info."""
        doc = self.analyses.find_one({"_id": ObjectId(analysis_id)})
        if doc:
            doc["id"] = str(doc.pop("_id"))
            # Get patient info
            patient = self.get_patient(doc.get("patient_id"))
            if patient:
                doc["patient"] = patient
        return doc

    def get_latest_analysis_for_ecg(self, ecg_record_id):
        doc = self.analyses.find_one(
            {"ecg_record_id": ecg_record_id},
            sort=[("created_at", -1)]
        )
        if doc is None:
            return None
        doc["id"] = str(doc.pop("_id"))
        return doc

    def get_history_for_user(self, user_id, limit=50):
        """
        Get analysis history for a user with patient info.
        """
        docs = list(self.analyses.find(
            {"user_id": user_id}
        ).sort("created_at", -1).limit(limit))
        
        for doc in docs:
            doc["id"] = str(doc.pop("_id"))
            # Get patient info
            patient = self.get_patient(doc.get("patient_id"))
            if patient:
                doc["patient"] = patient
        return docs

    # ============================================================
    # REPORTS
    # ============================================================

    def create_report(self, user_id, ecg_record_id, analysis_id, report_metadata,
                       report_file_path):
        doc = {
            "user_id": user_id,
            "ecg_record_id": ecg_record_id,
            "analysis_id": analysis_id,
            "report_metadata": report_metadata,
            "report_file_path": report_file_path,
            "created_at": self.now(),
        }
        result = self.reports.insert_one(doc)
        return str(result.inserted_id)

    def get_reports_for_user(self, user_id):
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