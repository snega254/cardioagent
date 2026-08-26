"""
Patient Context — Symptoms, History, Vitals Processing
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class PatientContext:
    """Patient information entered by user."""
    age: Optional[int] = None
    sex: Optional[str] = None
    symptoms: Optional[str] = None
    history: Optional[str] = None
    vitals: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "age": self.age,
            "sex": self.sex,
            "symptoms": self.symptoms,
            "history": self.history,
            "vitals": self.vitals or {}
        }
    
    def has_emergency_symptoms(self) -> bool:
        """
        Check if patient has emergency symptoms that require immediate attention.
        Returns True if any emergency symptom keywords are found.
        """
        if not self.symptoms:
            return False
        
        emergency_keywords = [
            "chest pain",
            "chest discomfort",
            "shortness of breath",
            "difficulty breathing",
            "severe dizziness",
            "fainting",
            "loss of consciousness",
            "palpitations",
            "sweating",
            "nausea with chest pain",
            "crushing chest pain",
            "unable to breathe",
            "severe chest pain",
            "tightness in chest",
            "pressure in chest"
        ]
        
        symptom_lower = self.symptoms.lower()
        return any(keyword in symptom_lower for keyword in emergency_keywords)
    
    def get_urgency_indicator(self) -> str:
        """Get urgency indicator based on symptoms."""
        if self.has_emergency_symptoms():
            return "⚠️ Emergency symptoms reported — seek immediate professional care"
        return ""


def parse_symptoms(symptom_text: str) -> Dict[str, Any]:
    """
    Parse symptom text into structured format.
    
    Args:
        symptom_text: Raw symptom text from user input
        
    Returns:
        Dictionary with parsed symptoms, severity, and raw text
    """
    if not symptom_text:
        return {"symptoms": [], "severity": None, "raw": ""}
    
    symptom_lower = symptom_text.lower()
    
    common_symptoms = [
        "chest pain", "chest discomfort", "shortness of breath", "palpitations",
        "dizziness", "fainting", "fatigue", "nausea", "sweating",
        "lightheadedness", "syncope", "presyncope", "dyspnea",
        "difficulty breathing", "tiredness", "weakness", "anxiety"
    ]
    
    detected = [s for s in common_symptoms if s in symptom_lower]
    
    # Severity indicators
    severity_level = "unknown"
    if any(w in symptom_lower for w in ["severe", "crushing", "unbearable", "worst", "extreme", "intense"]):
        severity_level = "severe"
    elif any(w in symptom_lower for w in ["moderate", "some", "mild", "slight"]):
        severity_level = "moderate"
    elif any(w in symptom_lower for w in ["mild", "minimal", "slight"]):
        severity_level = "mild"
    
    return {
        "symptoms": detected,
        "severity": severity_level,
        "raw": symptom_text,
        "has_emergency": any(
            keyword in symptom_lower 
            for keyword in ["chest pain", "shortness of breath", "difficulty breathing", 
                          "fainting", "loss of consciousness", "severe dizziness"]
        )
    }


def has_emergency_symptoms_from_text(symptom_text: str) -> bool:
    """Check if symptom text contains emergency keywords."""
    if not symptom_text:
        return False
    
    emergency_keywords = [
        "chest pain", "shortness of breath", "difficulty breathing",
        "fainting", "loss of consciousness", "severe dizziness",
        "palpitations", "sweating", "nausea with chest pain"
    ]
    
    symptom_lower = symptom_text.lower()
    return any(keyword in symptom_lower for keyword in emergency_keywords)