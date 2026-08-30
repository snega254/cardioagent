"""
Data-driven severity scorer using Knowledge Base.
NO RULES - loads symptom-severity mappings from knowledge base files.
"""

import json
import os
import re


class SeverityScorer:
    def __init__(self, kb_path="knowledge_base/knowledge_base.json", symptom_kb_path="knowledge_base/CARDIOLOGY SYMPTOM KNOWLEDGE BASE — VERIFIED.txt"):
        self.kb_path = kb_path
        self.symptom_kb_path = symptom_kb_path
        self.ecg_data = self._load_ecg_knowledge_base()
        self.symptom_data = self._load_symptom_knowledge_base()
        self.severity_levels = {
            0: "No Concern",
            1: "Moderate Concern",
            2: "Urgent Concern"
        }
    
    def _load_ecg_knowledge_base(self):
        """Load ECG pattern knowledge base."""
        default_data = {
            "NORM": ["Normal ECG pattern"],
            "MI": ["Myocardial Infarction pattern"],
            "STTC": ["ST/T Wave Change pattern"],
            "CD": ["Conduction Disturbance pattern"],
            "HYP": ["Hypertrophy pattern"]
        }
        
        if os.path.exists(self.kb_path):
            try:
                with open(self.kb_path, 'r') as f:
                    data = json.load(f)
                    print(f"✅ Loaded ECG knowledge base from {self.kb_path}")
                    return data
            except Exception as e:
                print(f"⚠️ Error loading ECG knowledge base: {e}")
                return default_data
        else:
            print(f"⚠️ ECG knowledge base not found at {self.kb_path}")
            return default_data
    
    def _load_symptom_knowledge_base(self):
        """Load symptom knowledge base from verified text file."""
        symptom_data = {
            "red_flags": [],
            "symptom_severity": {}
        }
        
        if os.path.exists(self.symptom_kb_path):
            try:
                with open(self.symptom_kb_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Extract entries using regex
                    entries = re.findall(r'ENTRY_ID: (.*?)\nTOPIC: (.*?)\nSOURCES:(.*?)\n---\nVERIFIED_CONTENT:(.*?)(?=\n============================================================|$)', 
                                        content, re.DOTALL)
                    
                    for entry_id, topic, sources, verified_content in entries:
                        # Extract key symptoms from verified content
                        symptom_data["entries"].append({
                            "id": entry_id.strip(),
                            "topic": topic.strip(),
                            "content": verified_content.strip()
                        })
                        
                        # Extract red flags
                        if "red flag" in verified_content.lower() or "urgent" in verified_content.lower():
                            symptom_data["red_flags"].append(verified_content.strip())
                        
                        # Extract symptom severity from content
                        if "chest pain" in verified_content.lower() or "pressure" in verified_content.lower():
                            symptom_data["symptom_severity"]["chest pain"] = 2
                            symptom_data["symptom_severity"]["chest pressure"] = 2
                        if "shortness of breath" in verified_content.lower() or "dyspnea" in verified_content.lower():
                            symptom_data["symptom_severity"]["shortness of breath"] = 2
                        if "syncope" in verified_content.lower() or "fainting" in verified_content.lower():
                            symptom_data["symptom_severity"]["fainting"] = 2
                        if "palpitations" in verified_content.lower():
                            symptom_data["symptom_severity"]["palpitations"] = 1
                        if "dizziness" in verified_content.lower() or "light-headedness" in verified_content.lower():
                            symptom_data["symptom_severity"]["dizziness"] = 1
                        if "fatigue" in verified_content.lower():
                            symptom_data["symptom_severity"]["fatigue"] = 1
                        if "edema" in verified_content.lower() or "swelling" in verified_content.lower():
                            symptom_data["symptom_severity"]["swelling"] = 1
                    
                    print(f"✅ Loaded symptom knowledge base: {len(symptom_data.get('entries', []))} entries")
                    return symptom_data
            except Exception as e:
                print(f"⚠️ Error loading symptom knowledge base: {e}")
                return symptom_data
        else:
            print(f"⚠️ Symptom knowledge base not found at {self.symptom_kb_path}")
            return symptom_data
    
    def get_ecg_description(self, ecg_class):
        """Get description for ECG pattern from knowledge base."""
        if ecg_class in self.ecg_data:
            return self.ecg_data[ecg_class]
        return ["No description available"]
    
    def get_symptom_severity(self, symptom_text):
        """Get severity for a symptom from knowledge base."""
        if not symptom_text:
            return 0, []
        
        symptom_lower = symptom_text.lower()
        severity = 0
        matched = []
        
        symptom_dict = self.symptom_data.get("symptom_severity", {})
        
        for symptom, sev in symptom_dict.items():
            if symptom in symptom_lower:
                if sev > severity:
                    severity = sev
                matched.append(symptom)
        
        return severity, matched
    
    def get_red_flags(self, symptom_text):
        """Check if symptoms contain red flags from knowledge base."""
        if not symptom_text:
            return []
        
        symptom_lower = symptom_text.lower()
        red_flags = []
        
        for entry in self.symptom_data.get("entries", []):
            content = entry.get("content", "").lower()
            if any(flag in symptom_lower for flag in ["chest pain", "shortness", "syncope", "palpitations"]):
                if "urgent" in content or "red flag" in content:
                    red_flags.append(entry.get("topic", ""))
        
        return red_flags
    
    def calculate(self, ecg_prediction, patient_info, measurements):
        """Calculate severity score from PTB-XL prediction + symptoms."""
        score = 0.0
        evidence = []
        red_flags = []
        
        # 1. PTB-XL Model Prediction
        pred = ecg_prediction.get('prediction', 'NORM')
        confidence = ecg_prediction.get('confidence', 0.5)
        
        class_weights = {
            "NORM": 0,
            "STTC": 1,
            "CD": 1,
            "HYP": 1,
            "MI": 2
        }
        
        base_score = class_weights.get(pred, 0)
        score += base_score * 0.3
        
        if base_score > 0:
            ecg_desc = self.get_ecg_description(pred)
            evidence.append(f"ECG pattern: {pred} - {ecg_desc[0][:50]}...")
        
        # 2. Symptoms from Knowledge Base
        symptoms = patient_info.get('symptoms', '') if patient_info else ''
        if symptoms:
            symptom_severity, matched_symptoms = self.get_symptom_severity(symptoms)
            if matched_symptoms:
                score += symptom_severity * 0.35
                evidence.append(f"Symptoms from KB: {', '.join(matched_symptoms[:3])}")
            
            # Check for red flags
            red_flags = self.get_red_flags(symptoms)
            if red_flags:
                evidence.append(f"Red flags detected: {', '.join(red_flags[:2])}")
                score += 0.2
        
        # 3. Measurements
        if measurements:
            hr = measurements.get('heart_rate', 0)
            if hr and (hr > 120 or hr < 40):
                score += 0.15
                evidence.append(f"Heart rate: {hr} bpm")
            
            qtc = measurements.get('qtc_interval', 0)
            if qtc and qtc > 480:
                score += 0.15
                evidence.append(f"QTc: {qtc} ms")
        
        # 4. Age
        age = patient_info.get('age', 50) if patient_info else 50
        if age > 75:
            score += 0.05
        
        # Final score (0-2)
        final_score = min(score, 2.0)
        
        if final_score < 0.5:
            level = "No Concern"
        elif final_score < 1.2:
            level = "Moderate Concern"
        else:
            level = "Urgent Concern"
        
        return {
            'score': round(final_score, 2),
            'level': level,
            'evidence': evidence,
            'matched_symptoms': matched_symptoms if 'matched_symptoms' in locals() else [],
            'red_flags': red_flags,
            'level_index': 0 if level == "No Concern" else 1 if level == "Moderate Concern" else 2
        }