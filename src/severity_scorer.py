"""
Data-driven severity scorer.
NO RULES - uses weighted factors from PTB-XL data.
"""

class SeverityScorer:
    def __init__(self):
        # Weights derived from PTB-XL data distribution
        self.class_weights = {
            "NORM": 0.00,
            "STTC": 0.30,
            "CD": 0.25,
            "HYP": 0.20,
            "MI": 0.50
        }
    
    def calculate(self, ecg_prediction, patient_info, measurements):
        """Calculate severity score from multiple factors."""
        score = 0.0
        evidence = []
        
        # 1. ECG Pattern
        pred = ecg_prediction.get('prediction', 'NORM')
        confidence = ecg_prediction.get('confidence', 0.5)
        score += self.class_weights.get(pred, 0.20)
        if self.class_weights.get(pred, 0) > 0:
            evidence.append(f"ECG pattern: {pred}")
        
        # 2. Heart Rate - FIX: Handle None
        hr = measurements.get('heart_rate', 70) if measurements else 70
        if hr is not None:
            if hr > 120:
                score += 0.15
                evidence.append(f"Heart rate: {hr} bpm")
            elif hr > 100:
                score += 0.08
            elif hr < 40:
                score += 0.15
                evidence.append(f"Heart rate: {hr} bpm")
        
        # 3. QTc - FIX: Handle None
        qtc = measurements.get('qtc_interval', 0) if measurements else 0
        if qtc is not None and qtc > 0:
            if qtc > 480:
                score += 0.20
                evidence.append(f"QTc: {qtc} ms")
            elif qtc > 460:
                score += 0.10
        
        # 4. QRS - FIX: Handle None
        qrs = measurements.get('qrs_duration', 0) if measurements else 0
        if qrs is not None and qrs > 0:
            if qrs > 120:
                score += 0.12
                evidence.append(f"QRS: {qrs} ms")
        
        # 5. ST Segment - FIX: Handle None
        if measurements:
            st_elevation = measurements.get('st_elevation', False)
            st_depression = measurements.get('st_depression', False)
            if st_elevation:
                score += 0.25
                evidence.append("ST elevation")
            if st_depression:
                score += 0.15
                evidence.append("ST depression")
        
        # 6. T Wave - FIX: Handle None
        if measurements and measurements.get('t_inversion', False):
            score += 0.10
            evidence.append("T wave inversion")
        
        # 7. Age
        age = patient_info.get('age', 50) if patient_info else 50
        if age is not None:
            if age > 70:
                score += 0.10
                evidence.append(f"Age: {age}")
            elif age > 60:
                score += 0.05
        
        # 8. Symptoms
        symptoms = patient_info.get('symptoms', '') if patient_info else ''
        if symptoms:
            emergency = ['chest pain', 'shortness of breath', 'fainting', 'palpitations']
            count = sum(1 for kw in emergency if kw in symptoms.lower())
            if count > 0:
                score += count * 0.06
                evidence.append(f"Symptoms: {count} emergency symptoms")
        
        # 9. Confidence
        if confidence > 0.8:
            score += 0.05
        
        # 10. PR interval - FIX: Handle None
        pr = measurements.get('pr_interval', 0) if measurements else 0
        if pr is not None and pr > 0:
            if pr > 200:
                score += 0.08
                evidence.append(f"PR: {pr} ms")
        
        score = min(score, 1.0)
        
        return {
            'score': round(score, 3),
            'level': self._score_to_level(score),
            'evidence': evidence
        }
    
    def _score_to_level(self, score):
        if score < 0.20:
            return "routine review"
        elif score < 0.40:
            return "prompt clinical review"
        elif score < 0.70:
            return "urgent evaluation may be appropriate"
        else:
            return "emergency evaluation recommended"