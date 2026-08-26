"""
Patient Management — Create, retrieve, and manage patients in MongoDB.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from bson import ObjectId


class PatientManager:
    """Manages patient records in MongoDB."""
    
    def __init__(self, db):
        """
        Args:
            db: CardioDB instance with MongoDB connection
        """
        self.db = db
        self.collection = db.db["patients"]
        self._ensure_indexes()
    
    def _ensure_indexes(self):
        """Create necessary indexes."""
        self.collection.create_index("user_id")
        self.collection.create_index("name")
    
    def create_patient(self, user_id: str, name: str, age: int, sex: str, symptoms: List[str]) -> str:
        """
        Create a new patient record.
        
        Args:
            user_id: ID of the logged-in user
            name: Patient's full name
            age: Patient's age
            sex: Patient's sex
            symptoms: List of symptoms
            
        Returns:
            patient_id: The new patient's ID
        """
        doc = {
            "user_id": user_id,
            "name": name.strip(),
            "age": age,
            "sex": sex,
            "symptoms": symptoms or [],
            "symptoms_text": ", ".join(symptoms) if symptoms else "",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        result = self.collection.insert_one(doc)
        return str(result.inserted_id)
    
    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get a patient by ID."""
        doc = self.collection.find_one({"_id": ObjectId(patient_id)})
        if doc:
            doc["id"] = str(doc.pop("_id"))
        return doc
    
    def get_patients_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all patients for a specific user."""
        docs = list(self.collection.find({"user_id": user_id}).sort("created_at", -1))
        for doc in docs:
            doc["id"] = str(doc.pop("_id"))
        return docs
    
    def update_patient(self, patient_id: str, **kwargs) -> bool:
        """Update patient information."""
        update_data = {k: v for k, v in kwargs.items() if v is not None}
        if not update_data:
            return False
        
        update_data["updated_at"] = datetime.utcnow().isoformat()
        
        # Handle symptoms special case
        if "symptoms" in update_data and isinstance(update_data["symptoms"], list):
            update_data["symptoms_text"] = ", ".join(update_data["symptoms"])
        
        result = self.collection.update_one(
            {"_id": ObjectId(patient_id)},
            {"$set": update_data}
        )
        return result.modified_count > 0
    
    def delete_patient(self, patient_id: str) -> bool:
        """Delete a patient and all associated analyses."""
        # Delete associated analyses first
        from db import CardioDB
        # Use the existing analyses collection
        analyses_collection = self.db.db["analyses"]
        analyses_collection.delete_many({"patient_id": patient_id})
        
        # Delete the patient
        result = self.collection.delete_one({"_id": ObjectId(patient_id)})
        return result.deleted_count > 0
    
    def search_patients(self, user_id: str, query: str) -> List[Dict[str, Any]]:
        """Search patients by name."""
        docs = list(self.collection.find({
            "user_id": user_id,
            "name": {"$regex": query, "$options": "i"}
        }).sort("created_at", -1))
        for doc in docs:
            doc["id"] = str(doc.pop("_id"))
        return docs


def get_patient_display_name(patient: Dict[str, Any]) -> str:
    """Get a display name for a patient."""
    name = patient.get("name", "Unknown")
    age = patient.get("age")
    sex = patient.get("sex", "")
    if age and sex:
        return f"{name} ({age}, {sex})"
    elif age:
        return f"{name} ({age})"
    elif sex:
        return f"{name} ({sex})"
    return name


def format_symptoms(symptoms: List[str]) -> str:
    """Format symptoms for display."""
    if not symptoms:
        return "No symptoms reported"
    return ", ".join(symptoms)