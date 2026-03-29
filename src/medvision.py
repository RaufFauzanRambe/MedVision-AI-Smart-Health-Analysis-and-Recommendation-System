#!/usr/bin/env python3
"""
================================================================================
MedVision – AI Smart Health Analysis and Recommendation System
================================================================================

A comprehensive AI-powered health analysis system that provides:
- Symptom analysis and disease prediction
- Health risk assessment
- Personalized health recommendations
- Health metrics tracking and visualization
- Medical report generation

Author: MedVision AI Team
Version: 1.0.0
License: MIT

DISCLAIMER: This system is for educational and informational purposes only.
Always consult a qualified healthcare professional for medical advice.
================================================================================
"""

import json
import datetime
import random
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from abc import ABC, abstractmethod
import statistics


# =============================================================================
# ENUMERATIONS AND CONSTANTS
# =============================================================================

class SeverityLevel(Enum):
    """Severity levels for health conditions."""
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"


class RiskCategory(Enum):
    """Risk categories for health assessments."""
    MINIMAL = "Minimal Risk"
    LOW = "Low Risk"
    MODERATE = "Moderate Risk"
    HIGH = "High Risk"
    VERY_HIGH = "Very High Risk"


class SymptomCategory(Enum):
    """Categories of symptoms."""
    GENERAL = "General"
    RESPIRATORY = "Respiratory"
    CARDIOVASCULAR = "Cardiovascular"
    DIGESTIVE = "Digestive"
    NEUROLOGICAL = "Neurological"
    MUSCULOSKELETAL = "Musculoskeletal"
    DERMATOLOGICAL = "Dermatological"
    PSYCHOLOGICAL = "Psychological"
    ENDOCRINE = "Endocrine"
    URINARY = "Urinary"


class AgeGroup(Enum):
    """Age group classifications."""
    CHILD = "Child (0-12)"
    ADOLESCENT = "Adolescent (13-17)"
    YOUNG_ADULT = "Young Adult (18-35)"
    MIDDLE_AGED = "Middle Aged (36-55)"
    SENIOR = "Senior (56-70)"
    ELDERLY = "Elderly (70+)"


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class Symptom:
    """Represents a medical symptom."""
    name: str
    category: SymptomCategory
    severity: SeverityLevel
    duration_days: int
    description: str = ""
    related_conditions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "category": self.category.value,
            "severity": self.severity.value,
            "duration_days": self.duration_days,
            "description": self.description,
            "related_conditions": self.related_conditions
        }


@dataclass
class VitalSigns:
    """Represents patient vital signs."""
    heart_rate: int  # bpm
    blood_pressure_systolic: int  # mmHg
    blood_pressure_diastolic: int  # mmHg
    temperature: float  # Celsius
    respiratory_rate: int  # breaths per minute
    oxygen_saturation: float  # percentage
    weight: float  # kg
    height: float  # cm
    blood_glucose: Optional[float] = None  # mg/dL
    cholesterol_total: Optional[float] = None  # mg/dL
    
    @property
    def bmi(self) -> float:
        """Calculate Body Mass Index."""
        height_m = self.height / 100
        return round(self.weight / (height_m ** 2), 2)
    
    @property
    def bmi_category(self) -> str:
        """Determine BMI category."""
        bmi = self.bmi
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"
    
    @property
    def blood_pressure_category(self) -> str:
        """Determine blood pressure category."""
        sys = self.blood_pressure_systolic
        dia = self.blood_pressure_diastolic
        
        if sys < 120 and dia < 80:
            return "Normal"
        elif sys < 130 and dia < 80:
            return "Elevated"
        elif sys < 140 or dia < 90:
            return "High Blood Pressure Stage 1"
        elif sys >= 140 or dia >= 90:
            return "High Blood Pressure Stage 2"
        elif sys > 180 or dia > 120:
            return "Hypertensive Crisis"
        return "Unknown"
    
    def to_dict(self) -> Dict:
        return {
            "heart_rate": self.heart_rate,
            "blood_pressure_systolic": self.blood_pressure_systolic,
            "blood_pressure_diastolic": self.blood_pressure_diastolic,
            "temperature": self.temperature,
            "respiratory_rate": self.respiratory_rate,
            "oxygen_saturation": self.oxygen_saturation,
            "weight": self.weight,
            "height": self.height,
            "bmi": self.bmi,
            "bmi_category": self.bmi_category,
            "blood_pressure_category": self.blood_pressure_category,
            "blood_glucose": self.blood_glucose,
            "cholesterol_total": self.cholesterol_total
        }


@dataclass
class PatientProfile:
    """Represents a patient profile."""
    patient_id: str
    name: str
    age: int
    gender: str
    blood_type: str = "Unknown"
    medical_history: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    current_medications: List[str] = field(default_factory=list)
    lifestyle_factors: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def age_group(self) -> AgeGroup:
        """Determine age group."""
        if self.age <= 12:
            return AgeGroup.CHILD
        elif self.age <= 17:
            return AgeGroup.ADOLESCENT
        elif self.age <= 35:
            return AgeGroup.YOUNG_ADULT
        elif self.age <= 55:
            return AgeGroup.MIDDLE_AGED
        elif self.age <= 70:
            return AgeGroup.SENIOR
        else:
            return AgeGroup.ELDERLY
    
    def to_dict(self) -> Dict:
        return {
            "patient_id": self.patient_id,
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "blood_type": self.blood_type,
            "medical_history": self.medical_history,
            "allergies": self.allergies,
            "current_medications": self.current_medications,
            "lifestyle_factors": self.lifestyle_factors,
            "age_group": self.age_group.value
        }


@dataclass
class HealthRecommendation:
    """Represents a health recommendation."""
    category: str
    title: str
    description: str
    priority: SeverityLevel
    actionable_steps: List[str]
    follow_up_timeline: str
    related_resources: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "actionable_steps": self.actionable_steps,
            "follow_up_timeline": self.follow_up_timeline,
            "related_resources": self.related_resources
        }


@dataclass
class RiskAssessment:
    """Represents a risk assessment result."""
    category: str
    risk_level: RiskCategory
    score: float  # 0-100
    factors: List[str]
    mitigation_strategies: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "category": self.category,
            "risk_level": self.risk_level.value,
            "score": self.score,
            "factors": self.factors,
            "mitigation_strategies": self.mitigation_strategies
        }


@dataclass
class DiagnosisResult:
    """Represents a potential diagnosis result."""
    condition: str
    confidence_score: float  # 0-100
    matching_symptoms: List[str]
    recommended_tests: List[str]
    urgency_level: SeverityLevel
    description: str
    
    def to_dict(self) -> Dict:
        return {
            "condition": self.condition,
            "confidence_score": self.confidence_score,
            "matching_symptoms": self.matching_symptoms,
            "recommended_tests": self.recommended_tests,
            "urgency_level": self.urgency_level.value,
            "description": self.description
        }


# =============================================================================
# MEDICAL KNOWLEDGE BASE
# =============================================================================

class MedicalKnowledgeBase:
    """
    A comprehensive medical knowledge base containing:
    - Disease-symptom relationships
    - Risk factors
    - Treatment protocols
    - Medical guidelines
    """
    
    # Disease database with symptoms and risk factors
    DISEASES = {
        "Common Cold": {
            "symptoms": ["runny nose", "sneezing", "sore throat", "cough", "mild fever", 
                        "nasal congestion", "headache", "fatigue"],
            "risk_factors": ["weakened immune system", "age over 65", "age under 5"],
            "severity": SeverityLevel.LOW,
            "description": "A viral infection of the upper respiratory tract",
            "recommended_tests": ["Physical examination"],
            "categories": [SymptomCategory.RESPIRATORY, SymptomCategory.GENERAL]
        },
        "Influenza": {
            "symptoms": ["high fever", "body aches", "fatigue", "cough", "sore throat",
                        "headache", "chills", "runny nose"],
            "risk_factors": ["age over 65", "age under 5", "pregnancy", "chronic diseases"],
            "severity": SeverityLevel.MODERATE,
            "description": "A contagious respiratory illness caused by influenza viruses",
            "recommended_tests": ["Rapid influenza diagnostic test", "PCR test"],
            "categories": [SymptomCategory.RESPIRATORY, SymptomCategory.GENERAL]
        },
        "Hypertension": {
            "symptoms": ["headache", "dizziness", "blurred vision", "shortness of breath",
                        "chest pain", "irregular heartbeat"],
            "risk_factors": ["family history", "obesity", "sedentary lifestyle", "high salt diet",
                           "age over 65", "stress", "smoking", "alcohol consumption"],
            "severity": SeverityLevel.HIGH,
            "description": "High blood pressure that can lead to serious health problems",
            "recommended_tests": ["Blood pressure monitoring", "ECG", "Blood tests"],
            "categories": [SymptomCategory.CARDIOVASCULAR]
        },
        "Type 2 Diabetes": {
            "symptoms": ["increased thirst", "frequent urination", "increased hunger",
                        "unexplained weight loss", "fatigue", "blurred vision", "slow healing sores"],
            "risk_factors": ["obesity", "family history", "age over 45", "sedentary lifestyle",
                           "prediabetes", "gestational diabetes history"],
            "severity": SeverityLevel.HIGH,
            "description": "A chronic condition affecting how the body processes blood sugar",
            "recommended_tests": ["Fasting blood glucose", "HbA1c test", "Oral glucose tolerance test"],
            "categories": [SymptomCategory.ENDOCRINE, SymptomCategory.GENERAL]
        },
        "Anxiety Disorder": {
            "symptoms": ["excessive worry", "restlessness", "fatigue", "difficulty concentrating",
                        "irritability", "muscle tension", "sleep problems", "rapid heartbeat"],
            "risk_factors": ["family history", "trauma", "stress", "chronic illness",
                           "substance abuse", "personality type"],
            "severity": SeverityLevel.MODERATE,
            "description": "A mental health disorder characterized by feelings of worry and fear",
            "recommended_tests": ["Psychological evaluation", "GAD-7 questionnaire"],
            "categories": [SymptomCategory.PSYCHOLOGICAL]
        },
        "Migraine": {
            "symptoms": ["severe headache", "nausea", "vomiting", "sensitivity to light",
                        "sensitivity to sound", "visual disturbances", "dizziness"],
            "risk_factors": ["family history", "hormonal changes", "stress", "certain foods",
                           "sleep changes", "environmental factors"],
            "severity": SeverityLevel.MODERATE,
            "description": "A neurological condition causing intense, debilitating headaches",
            "recommended_tests": ["Neurological examination", "MRI", "CT scan"],
            "categories": [SymptomCategory.NEUROLOGICAL]
        },
        "Gastroesophageal Reflux Disease (GERD)": {
            "symptoms": ["heartburn", "chest pain", "difficulty swallowing", "regurgitation",
                        "chronic cough", "sore throat", "feeling of lump in throat"],
            "risk_factors": ["obesity", "hiatal hernia", "pregnancy", "smoking",
                           "certain medications", "late meals"],
            "severity": SeverityLevel.MODERATE,
            "description": "A chronic digestive disease where stomach acid flows back into the esophagus",
            "recommended_tests": ["Endoscopy", "Ambulatory acid probe test", "Esophageal manometry"],
            "categories": [SymptomCategory.DIGESTIVE]
        },
        "Pneumonia": {
            "symptoms": ["chest pain", "cough with phlegm", "fatigue", "fever", "shortness of breath",
                        "sweating", "shaking chills", "lower body temperature"],
            "risk_factors": ["age over 65", "age under 2", "weakened immune system",
                           "chronic diseases", "smoking", "hospitalization"],
            "severity": SeverityLevel.HIGH,
            "description": "An infection that inflames the air sacs in one or both lungs",
            "recommended_tests": ["Chest X-ray", "Blood tests", "Sputum test", "CT scan"],
            "categories": [SymptomCategory.RESPIRATORY]
        },
        "Arthritis": {
            "symptoms": ["joint pain", "joint stiffness", "swelling", "redness around joints",
                        "decreased range of motion", "fatigue"],
            "risk_factors": ["age", "family history", "obesity", "joint injury",
                           "occupation", "autoimmune conditions"],
            "severity": SeverityLevel.MODERATE,
            "description": "Inflammation of one or more joints causing pain and stiffness",
            "recommended_tests": ["X-ray", "MRI", "Blood tests", "Joint fluid analysis"],
            "categories": [SymptomCategory.MUSCULOSKELETAL]
        },
        "Urinary Tract Infection": {
            "symptoms": ["burning sensation when urinating", "frequent urination", "cloudy urine",
                        "strong-smelling urine", "pelvic pain", "blood in urine"],
            "risk_factors": ["female anatomy", "sexual activity", "certain birth control",
                           "menopause", "urinary tract abnormalities", "blocked urinary tract"],
            "severity": SeverityLevel.MODERATE,
            "description": "An infection in any part of the urinary system",
            "recommended_tests": ["Urinalysis", "Urine culture", "CT scan", "Cystoscopy"],
            "categories": [SymptomCategory.URINARY]
        }
    }
    
    # Symptom severity weights
    SYMPTOM_SEVERITY_WEIGHTS = {
        SeverityLevel.LOW: 0.25,
        SeverityLevel.MODERATE: 0.50,
        SeverityLevel.HIGH: 0.75,
        SeverityLevel.CRITICAL: 1.00
    }
    
    # Vital signs normal ranges
    VITAL_SIGN_RANGES = {
        "heart_rate": {"min": 60, "max": 100, "unit": "bpm"},
        "blood_pressure_systolic": {"min": 90, "max": 120, "unit": "mmHg"},
        "blood_pressure_diastolic": {"min": 60, "max": 80, "unit": "mmHg"},
        "temperature": {"min": 36.1, "max": 37.2, "unit": "°C"},
        "respiratory_rate": {"min": 12, "max": 20, "unit": "breaths/min"},
        "oxygen_saturation": {"min": 95, "max": 100, "unit": "%"},
        "blood_glucose_fasting": {"min": 70, "max": 100, "unit": "mg/dL"},
        "blood_glucose_random": {"min": 70, "max": 140, "unit": "mg/dL"},
        "cholesterol_total": {"min": 0, "max": 200, "unit": "mg/dL"}
    }
    
    # Lifestyle recommendations
    LIFESTYLE_RECOMMENDATIONS = {
        "exercise": {
            "title": "Regular Physical Activity",
            "benefits": ["improved cardiovascular health", "weight management", 
                        "mental health improvement", "increased energy"],
            "guidelines": "Aim for at least 150 minutes of moderate-intensity exercise per week"
        },
        "nutrition": {
            "title": "Balanced Nutrition",
            "benefits": ["disease prevention", "energy maintenance", "weight control",
                        "immune system support"],
            "guidelines": "Eat a variety of fruits, vegetables, whole grains, and lean proteins"
        },
        "sleep": {
            "title": "Quality Sleep",
            "benefits": ["cognitive function", "immune health", "mood regulation",
                        "physical recovery"],
            "guidelines": "Aim for 7-9 hours of quality sleep each night"
        },
        "hydration": {
            "title": "Proper Hydration",
            "benefits": ["digestion", "temperature regulation", "cognitive function",
                        "joint health"],
            "guidelines": "Drink at least 8 glasses of water daily"
        },
        "stress_management": {
            "title": "Stress Management",
            "benefits": ["mental health", "cardiovascular health", "immune function",
                        "sleep quality"],
            "guidelines": "Practice relaxation techniques, mindfulness, or meditation daily"
        }
    }
    
    @classmethod
    def get_disease_by_symptom(cls, symptom_name: str) -> List[Dict]:
        """Find diseases associated with a symptom."""
        matching_diseases = []
        symptom_lower = symptom_name.lower()
        
        for disease_name, disease_info in cls.DISEASES.items():
            for symptom in disease_info["symptoms"]:
                if symptom_lower in symptom.lower() or symptom.lower() in symptom_lower:
                    matching_diseases.append({
                        "disease": disease_name,
                        "matching_symptom": symptom,
                        "severity": disease_info["severity"],
                        "categories": [c.value for c in disease_info["categories"]]
                    })
                    break
        
        return matching_diseases
    
    @classmethod
    def get_all_symptoms(cls) -> List[str]:
        """Get all unique symptoms in the knowledge base."""
        all_symptoms = set()
        for disease_info in cls.DISEASES.values():
            all_symptoms.update(disease_info["symptoms"])
        return sorted(list(all_symptoms))


# =============================================================================
# ANALYSIS ENGINES
# =============================================================================

class SymptomAnalyzer:
    """Analyzes symptoms and predicts potential conditions."""
    
    def __init__(self, knowledge_base: MedicalKnowledgeBase):
        self.kb = knowledge_base
    
    def analyze_symptoms(self, symptoms: List[Symptom]) -> List[DiagnosisResult]:
        """
        Analyze a list of symptoms and return potential diagnoses.
        Uses a scoring algorithm based on symptom matching and severity.
        """
        results = []
        
        # Extract symptom names (lowercase for matching)
        symptom_names = [s.name.lower() for s in symptoms]
        symptom_severities = {s.name.lower(): s.severity for s in symptoms}
        
        for disease_name, disease_info in self.kb.DISEASES.items():
            matching_symptoms = []
            match_score = 0
            
            for disease_symptom in disease_info["symptoms"]:
                for input_symptom in symptom_names:
                    if disease_symptom.lower() in input_symptom or input_symptom in disease_symptom.lower():
                        matching_symptoms.append(disease_symptom)
                        # Weight by symptom severity
                        severity = symptom_severities.get(input_symptom, SeverityLevel.MODERATE)
                        weight = self.kb.SYMPTOM_SEVERITY_WEIGHTS[severity]
                        match_score += weight
                        break
            
            if matching_symptoms:
                # Calculate confidence score (percentage of disease symptoms matched)
                coverage_score = len(matching_symptoms) / len(disease_info["symptoms"]) * 100
                # Combine with match score
                confidence = (match_score / len(disease_info["symptoms"])) * 50 + coverage_score * 0.5
                confidence = min(100, confidence)
                
                # Determine urgency based on severity and confidence
                if disease_info["severity"] == SeverityLevel.CRITICAL:
                    urgency = SeverityLevel.CRITICAL
                elif disease_info["severity"] == SeverityLevel.HIGH and confidence > 50:
                    urgency = SeverityLevel.HIGH
                elif confidence > 30:
                    urgency = SeverityLevel.MODERATE
                else:
                    urgency = SeverityLevel.LOW
                
                result = DiagnosisResult(
                    condition=disease_name,
                    confidence_score=round(confidence, 2),
                    matching_symptoms=matching_symptoms,
                    recommended_tests=disease_info["recommended_tests"],
                    urgency_level=urgency,
                    description=disease_info["description"]
                )
                results.append(result)
        
        # Sort by confidence score
        results.sort(key=lambda x: x.confidence_score, reverse=True)
        return results
    
    def get_symptom_suggestions(self, partial_symptom: str) -> List[str]:
        """Get symptom suggestions based on partial input."""
        all_symptoms = self.kb.get_all_symptoms()
        partial_lower = partial_symptom.lower()
        return [s for s in all_symptoms if partial_lower in s.lower()]


class VitalSignsAnalyzer:
    """Analyzes vital signs and identifies abnormalities."""
    
    def __init__(self, knowledge_base: MedicalKnowledgeBase):
        self.kb = knowledge_base
    
    def analyze_vital_signs(self, vital_signs: VitalSigns, patient: PatientProfile) -> Dict[str, Any]:
        """
        Analyze vital signs and return assessment with recommendations.
        """
        analysis = {
            "timestamp": datetime.datetime.now().isoformat(),
            "overall_status": "Normal",
            "abnormalities": [],
            "recommendations": [],
            "risk_factors": []
        }
        
        # Age-adjusted thresholds
        age = patient.age
        age_factor = 1 + (age - 30) * 0.005 if age > 30 else 1
        
        # Heart Rate Analysis
        hr = vital_signs.heart_rate
        hr_range = self.kb.VITAL_SIGN_RANGES["heart_rate"]
        if hr < hr_range["min"]:
            analysis["abnormalities"].append({
                "parameter": "Heart Rate",
                "value": f"{hr} {hr_range['unit']}",
                "status": "Low (Bradycardia)",
                "normal_range": f"{hr_range['min']}-{hr_range['max']} {hr_range['unit']}"
            })
            analysis["overall_status"] = "Attention Required"
        elif hr > hr_range["max"] * age_factor:
            analysis["abnormalities"].append({
                "parameter": "Heart Rate",
                "value": f"{hr} {hr_range['unit']}",
                "status": "High (Tachycardia)",
                "normal_range": f"{hr_range['min']}-{hr_range['max']} {hr_range['unit']}"
            })
            analysis["overall_status"] = "Attention Required"
        
        # Blood Pressure Analysis
        sys = vital_signs.blood_pressure_systolic
        dia = vital_signs.blood_pressure_diastolic
        bp_category = vital_signs.blood_pressure_category
        
        if bp_category != "Normal":
            analysis["abnormalities"].append({
                "parameter": "Blood Pressure",
                "value": f"{sys}/{dia} mmHg",
                "status": bp_category,
                "normal_range": "<120/80 mmHg"
            })
            if "High" in bp_category or "Crisis" in bp_category:
                analysis["overall_status"] = "Medical Attention Recommended"
        
        # Temperature Analysis
        temp = vital_signs.temperature
        temp_range = self.kb.VITAL_SIGN_RANGES["temperature"]
        if temp > temp_range["max"]:
            fever_severity = "Low-grade fever" if temp < 38 else "Fever" if temp < 39 else "High fever"
            analysis["abnormalities"].append({
                "parameter": "Temperature",
                "value": f"{temp} {temp_range['unit']}",
                "status": fever_severity,
                "normal_range": f"{temp_range['min']}-{temp_range['max']} {temp_range['unit']}"
            })
        elif temp < temp_range["min"]:
            analysis["abnormalities"].append({
                "parameter": "Temperature",
                "value": f"{temp} {temp_range['unit']}",
                "status": "Hypothermia",
                "normal_range": f"{temp_range['min']}-{temp_range['max']} {temp_range['unit']}"
            })
        
        # Respiratory Rate Analysis
        rr = vital_signs.respiratory_rate
        rr_range = self.kb.VITAL_SIGN_RANGES["respiratory_rate"]
        if rr < rr_range["min"]:
            analysis["abnormalities"].append({
                "parameter": "Respiratory Rate",
                "value": f"{rr} {rr_range['unit']}",
                "status": "Low (Bradypnea)",
                "normal_range": f"{rr_range['min']}-{rr_range['max']} {rr_range['unit']}"
            })
        elif rr > rr_range["max"]:
            analysis["abnormalities"].append({
                "parameter": "Respiratory Rate",
                "value": f"{rr} {rr_range['unit']}",
                "status": "High (Tachypnea)",
                "normal_range": f"{rr_range['min']}-{rr_range['max']} {rr_range['unit']}"
            })
        
        # Oxygen Saturation Analysis
        spo2 = vital_signs.oxygen_saturation
        spo2_range = self.kb.VITAL_SIGN_RANGES["oxygen_saturation"]
        if spo2 < spo2_range["min"]:
            analysis["abnormalities"].append({
                "parameter": "Oxygen Saturation",
                "value": f"{spo2} {spo2_range['unit']}",
                "status": "Low (Hypoxemia)" if spo2 < 90 else "Borderline Low",
                "normal_range": f"{spo2_range['min']}-{spo2_range['max']} {spo2_range['unit']}"
            })
            if spo2 < 90:
                analysis["overall_status"] = "Urgent Medical Attention Required"
        
        # BMI Analysis
        bmi_category = vital_signs.bmi_category
        if bmi_category != "Normal":
            analysis["abnormalities"].append({
                "parameter": "BMI",
                "value": f"{vital_signs.bmi} kg/m²",
                "status": bmi_category,
                "normal_range": "18.5-24.9 kg/m²"
            })
        
        # Blood Glucose Analysis (if available)
        if vital_signs.blood_glucose is not None:
            glucose = vital_signs.blood_glucose
            glucose_range = self.kb.VITAL_SIGN_RANGES["blood_glucose_fasting"]
            if glucose > glucose_range["max"]:
                status = "Prediabetes range" if glucose < 126 else "Diabetes range"
                analysis["abnormalities"].append({
                    "parameter": "Blood Glucose",
                    "value": f"{glucose} {glucose_range['unit']}",
                    "status": status,
                    "normal_range": f"{glucose_range['min']}-{glucose_range['max']} {glucose_range['unit']} (fasting)"
                })
        
        # Cholesterol Analysis (if available)
        if vital_signs.cholesterol_total is not None:
            chol = vital_signs.cholesterol_total
            chol_range = self.kb.VITAL_SIGN_RANGES["cholesterol_total"]
            if chol > chol_range["max"]:
                status = "Borderline high" if chol < 240 else "High"
                analysis["abnormalities"].append({
                    "parameter": "Total Cholesterol",
                    "value": f"{chol} {chol_range['unit']}",
                    "status": status,
                    "normal_range": f"<{chol_range['max']} {chol_range['unit']}"
                })
        
        # Generate recommendations based on analysis
        analysis["recommendations"] = self._generate_recommendations(analysis, vital_signs, patient)
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict, vital_signs: VitalSigns, 
                                   patient: PatientProfile) -> List[HealthRecommendation]:
        """Generate personalized recommendations based on vital signs analysis."""
        recommendations = []
        
        for abnormality in analysis["abnormalities"]:
            param = abnormality["parameter"]
            status = abnormality["status"]
            
            if param == "Blood Pressure" and "High" in status:
                recommendations.append(HealthRecommendation(
                    category="Cardiovascular",
                    title="Blood Pressure Management",
                    description=f"Your blood pressure reading of {abnormality['value']} indicates {status}. "
                              "Elevated blood pressure can lead to serious cardiovascular complications if left unmanaged.",
                    priority=SeverityLevel.HIGH,
                    actionable_steps=[
                        "Monitor blood pressure daily and keep a log",
                        "Reduce sodium intake to less than 2,300mg/day",
                        "Engage in regular aerobic exercise (30 min/day)",
                        "Limit alcohol consumption",
                        "Consider stress reduction techniques",
                        "Consult a physician for medication evaluation"
                    ],
                    follow_up_timeline="Within 2 weeks",
                    related_resources=["American Heart Association", "DASH Diet Guidelines"]
                ))
            
            elif param == "Oxygen Saturation" and "Low" in status:
                recommendations.append(HealthRecommendation(
                    category="Respiratory",
                    title="Oxygen Saturation Concern",
                    description=f"Your oxygen saturation is {vital_signs.oxygen_saturation}%, which is below normal. "
                              "This requires immediate attention to identify the underlying cause.",
                    priority=SeverityLevel.CRITICAL if vital_signs.oxygen_saturation < 90 else SeverityLevel.HIGH,
                    actionable_steps=[
                        "Seek immediate medical attention if below 90%",
                        "Avoid exertion until evaluated",
                        "Note any associated symptoms (shortness of breath, chest pain)",
                        "Prepare for possible pulse oximetry monitoring and ABG test"
                    ],
                    follow_up_timeline="Immediate",
                    related_resources=["Pulmonology Specialist", "Emergency Care"]
                ))
            
            elif param == "BMI" and status in ["Overweight", "Obese"]:
                recommendations.append(HealthRecommendation(
                    category="Weight Management",
                    title="Weight Management Program",
                    description=f"Your BMI of {vital_signs.bmi} falls in the {status} category. "
                              "Achieving a healthy weight can significantly reduce health risks.",
                    priority=SeverityLevel.MODERATE,
                    actionable_steps=[
                        "Calculate daily caloric needs and create a moderate deficit",
                        "Increase physical activity gradually",
                        "Focus on whole foods and reduce processed food intake",
                        "Consider consulting a registered dietitian",
                        "Set realistic weight loss goals (1-2 lbs per week)"
                    ],
                    follow_up_timeline="Ongoing with 3-month review",
                    related_resources=["Nutrition Counseling", "Weight Management Programs"]
                ))
            
            elif param == "Blood Glucose" and "Diabetes" in status:
                recommendations.append(HealthRecommendation(
                    category="Endocrine",
                    title="Blood Glucose Management",
                    description=f"Your blood glucose level of {vital_signs.blood_glucose} mg/dL suggests "
                              "the need for further evaluation and possible diabetes management.",
                    priority=SeverityLevel.HIGH,
                    actionable_steps=[
                        "Schedule HbA1c test for long-term glucose assessment",
                        "Monitor blood glucose levels regularly",
                        "Adopt a low glycemic index diet",
                        "Increase physical activity",
                        "Consult an endocrinologist",
                        "Check for symptoms of diabetes (increased thirst, urination)"
                    ],
                    follow_up_timeline="Within 1 week",
                    related_resources=["Diabetes Education", "Endocrinology Specialist"]
                ))
        
        return recommendations


class RiskAssessmentEngine:
    """Comprehensive health risk assessment engine."""
    
    def __init__(self, knowledge_base: MedicalKnowledgeBase):
        self.kb = knowledge_base
    
    def assess_cardiovascular_risk(self, patient: PatientProfile, 
                                    vital_signs: VitalSigns) -> RiskAssessment:
        """Assess cardiovascular disease risk."""
        risk_factors = []
        risk_score = 0
        
        # Age factor
        if patient.age > 55:
            risk_factors.append(f"Age ({patient.age} years)")
            risk_score += 10
        if patient.age > 65:
            risk_score += 5
        
        # Blood pressure
        if vital_signs.blood_pressure_systolic > 140 or vital_signs.blood_pressure_diastolic > 90:
            risk_factors.append("High blood pressure")
            risk_score += 15
        elif vital_signs.blood_pressure_systolic > 130 or vital_signs.blood_pressure_diastolic > 85:
            risk_factors.append("Elevated blood pressure")
            risk_score += 8
        
        # BMI
        if vital_signs.bmi > 30:
            risk_factors.append(f"Obesity (BMI: {vital_signs.bmi})")
            risk_score += 12
        elif vital_signs.bmi > 25:
            risk_factors.append(f"Overweight (BMI: {vital_signs.bmi})")
            risk_score += 6
        
        # Cholesterol
        if vital_signs.cholesterol_total and vital_signs.cholesterol_total > 240:
            risk_factors.append(f"High cholesterol ({vital_signs.cholesterol_total} mg/dL)")
            risk_score += 10
        elif vital_signs.cholesterol_total and vital_signs.cholesterol_total > 200:
            risk_factors.append(f"Borderline high cholesterol ({vital_signs.cholesterol_total} mg/dL)")
            risk_score += 5
        
        # Blood glucose
        if vital_signs.blood_glucose and vital_signs.blood_glucose > 126:
            risk_factors.append(f"High blood glucose ({vital_signs.blood_glucose} mg/dL)")
            risk_score += 15
        elif vital_signs.blood_glucose and vital_signs.blood_glucose > 100:
            risk_factors.append(f"Prediabetes ({vital_signs.blood_glucose} mg/dL)")
            risk_score += 8
        
        # Lifestyle factors
        lifestyle = patient.lifestyle_factors
        if lifestyle.get("smoking", False):
            risk_factors.append("Current smoker")
            risk_score += 15
        if lifestyle.get("sedentary", False):
            risk_factors.append("Sedentary lifestyle")
            risk_score += 8
        if lifestyle.get("family_history_cv", False):
            risk_factors.append("Family history of cardiovascular disease")
            risk_score += 10
        
        # Medical history
        if "hypertension" in [h.lower() for h in patient.medical_history]:
            risk_factors.append("History of hypertension")
            risk_score += 10
        if "diabetes" in [h.lower() for h in patient.medical_history]:
            risk_factors.append("History of diabetes")
            risk_score += 12
        
        # Determine risk level
        risk_score = min(100, risk_score)
        if risk_score < 10:
            risk_level = RiskCategory.MINIMAL
        elif risk_score < 25:
            risk_level = RiskCategory.LOW
        elif risk_score < 50:
            risk_level = RiskCategory.MODERATE
        elif risk_score < 75:
            risk_level = RiskCategory.HIGH
        else:
            risk_level = RiskCategory.VERY_HIGH
        
        # Mitigation strategies
        mitigation = [
            "Regular cardiovascular screening",
            "Maintain healthy blood pressure",
            "Follow heart-healthy diet",
            "Regular physical activity (150 min/week)",
            "Maintain healthy weight",
            "Avoid smoking and limit alcohol",
            "Manage stress effectively"
        ]
        
        if risk_level in [RiskCategory.HIGH, RiskCategory.VERY_HIGH]:
            mitigation.insert(0, "Consult cardiologist for comprehensive evaluation")
        
        return RiskAssessment(
            category="Cardiovascular Disease",
            risk_level=risk_level,
            score=risk_score,
            factors=risk_factors,
            mitigation_strategies=mitigation
        )
    
    def assess_diabetes_risk(self, patient: PatientProfile, 
                             vital_signs: VitalSigns) -> RiskAssessment:
        """Assess Type 2 Diabetes risk."""
        risk_factors = []
        risk_score = 0
        
        # Age
        if patient.age > 45:
            risk_factors.append(f"Age over 45 ({patient.age} years)")
            risk_score += 10
        
        # BMI
        if vital_signs.bmi > 30:
            risk_factors.append(f"Obesity (BMI: {vital_signs.bmi})")
            risk_score += 20
        elif vital_signs.bmi > 25:
            risk_factors.append(f"Overweight (BMI: {vital_signs.bmi})")
            risk_score += 10
        
        # Blood glucose
        if vital_signs.blood_glucose:
            if vital_signs.blood_glucose > 126:
                risk_factors.append(f"Fasting glucose in diabetes range ({vital_signs.blood_glucose} mg/dL)")
                risk_score += 25
            elif vital_signs.blood_glucose > 100:
                risk_factors.append(f"Impaired fasting glucose ({vital_signs.blood_glucose} mg/dL)")
                risk_score += 15
        
        # Lifestyle
        lifestyle = patient.lifestyle_factors
        if lifestyle.get("sedentary", False):
            risk_factors.append("Physical inactivity")
            risk_score += 10
        
        # Family history
        if lifestyle.get("family_history_diabetes", False):
            risk_factors.append("Family history of diabetes")
            risk_score += 15
        
        # Medical history
        if "gestational_diabetes" in [h.lower() for h in patient.medical_history]:
            risk_factors.append("History of gestational diabetes")
            risk_score += 15
        if "pcos" in [h.lower() for h in patient.medical_history]:
            risk_factors.append("Polycystic ovary syndrome")
            risk_score += 10
        
        # Blood pressure
        if vital_signs.blood_pressure_systolic > 140 or vital_signs.blood_pressure_diastolic > 90:
            risk_factors.append("Hypertension")
            risk_score += 10
        
        # Determine risk level
        risk_score = min(100, risk_score)
        if risk_score < 10:
            risk_level = RiskCategory.MINIMAL
        elif risk_score < 25:
            risk_level = RiskCategory.LOW
        elif risk_score < 50:
            risk_level = RiskCategory.MODERATE
        elif risk_score < 75:
            risk_level = RiskCategory.HIGH
        else:
            risk_level = RiskCategory.VERY_HIGH
        
        mitigation = [
            "Regular blood glucose screening",
            "Maintain healthy weight",
            "Follow low glycemic index diet",
            "Regular physical activity",
            "Limit refined carbohydrates and sugars",
            "Monitor for symptoms (increased thirst, urination)"
        ]
        
        if risk_level in [RiskCategory.MODERATE, RiskCategory.HIGH, RiskCategory.VERY_HIGH]:
            mitigation.append("Consult physician for HbA1c testing")
        
        return RiskAssessment(
            category="Type 2 Diabetes",
            risk_level=risk_level,
            score=risk_score,
            factors=risk_factors,
            mitigation_strategies=mitigation
        )
    
    def comprehensive_risk_assessment(self, patient: PatientProfile, 
                                       vital_signs: VitalSigns) -> List[RiskAssessment]:
        """Perform comprehensive risk assessment across multiple categories."""
        assessments = [
            self.assess_cardiovascular_risk(patient, vital_signs),
            self.assess_diabetes_risk(patient, vital_signs)
        ]
        
        # Add more assessments as needed
        # assessments.append(self.assess_cancer_risk(patient, vital_signs))
        
        # Sort by risk score
        assessments.sort(key=lambda x: x.score, reverse=True)
        return assessments


# =============================================================================
# RECOMMENDATION ENGINE
# =============================================================================

class AIRecommendationEngine:
    """
    AI-powered recommendation engine that provides personalized health recommendations
    based on patient profile, symptoms, vital signs, and risk assessments.
    """
    
    def __init__(self, knowledge_base: MedicalKnowledgeBase):
        self.kb = knowledge_base
    
    def generate_recommendations(self, patient: PatientProfile, 
                                  vital_signs: VitalSigns,
                                  symptoms: List[Symptom],
                                  risk_assessments: List[RiskAssessment],
                                  diagnosis_results: List[DiagnosisResult]) -> List[HealthRecommendation]:
        """Generate comprehensive personalized recommendations."""
        recommendations = []
        
        # Add symptom-based recommendations
        recommendations.extend(self._symptom_based_recommendations(symptoms))
        
        # Add vital signs-based recommendations
        recommendations.extend(self._vital_signs_recommendations(vital_signs, patient))
        
        # Add risk-based recommendations
        recommendations.extend(self._risk_based_recommendations(risk_assessments))
        
        # Add lifestyle recommendations
        recommendations.extend(self._lifestyle_recommendations(patient, vital_signs))
        
        # Add follow-up recommendations based on diagnoses
        recommendations.extend(self._diagnosis_followup_recommendations(diagnosis_results))
        
        # Remove duplicates and sort by priority
        unique_recommendations = self._deduplicate_recommendations(recommendations)
        unique_recommendations.sort(key=lambda x: self._priority_score(x.priority), reverse=True)
        
        return unique_recommendations
    
    def _priority_score(self, priority: SeverityLevel) -> int:
        """Convert priority to numeric score for sorting."""
        scores = {
            SeverityLevel.LOW: 1,
            SeverityLevel.MODERATE: 2,
            SeverityLevel.HIGH: 3,
            SeverityLevel.CRITICAL: 4
        }
        return scores.get(priority, 0)
    
    def _symptom_based_recommendations(self, symptoms: List[Symptom]) -> List[HealthRecommendation]:
        """Generate recommendations based on symptoms."""
        recommendations = []
        
        # Group symptoms by category
        category_symptoms = {}
        for symptom in symptoms:
            if symptom.category not in category_symptoms:
                category_symptoms[symptom.category] = []
            category_symptoms[symptom.category].append(symptom)
        
        # Generate category-specific recommendations
        for category, cat_symptoms in category_symptoms.items():
            high_severity = [s for s in cat_symptoms if s.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]]
            
            if category == SymptomCategory.RESPIRATORY:
                if high_severity:
                    recommendations.append(HealthRecommendation(
                        category="Respiratory Health",
                        title="Respiratory Symptoms Evaluation",
                        description=f"You have {len(high_severity)} severe respiratory symptom(s). "
                                  "These symptoms warrant prompt medical evaluation.",
                        priority=SeverityLevel.HIGH,
                        actionable_steps=[
                            "Schedule appointment with healthcare provider",
                            "Monitor oxygen saturation if available",
                            "Note any triggers or patterns",
                            "Stay hydrated and rest",
                            "Avoid irritants (smoke, dust, allergens)"
                        ],
                        follow_up_timeline="Within 48 hours",
                        related_resources=["Pulmonology", "Primary Care"]
                    ))
            
            elif category == SymptomCategory.CARDIOVASCULAR:
                recommendations.append(HealthRecommendation(
                    category="Cardiovascular Health",
                    title="Cardiovascular Symptom Assessment",
                    description="Cardiovascular symptoms require careful evaluation to rule out serious conditions.",
                    priority=SeverityLevel.HIGH if high_severity else SeverityLevel.MODERATE,
                    actionable_steps=[
                        "Seek immediate medical attention if chest pain is severe",
                        "Monitor blood pressure regularly",
                        "Reduce physical exertion until evaluated",
                        "Document symptoms and triggers",
                        "Avoid stimulants (caffeine, nicotine)"
                    ],
                    follow_up_timeline="Within 24 hours" if high_severity else "Within 1 week",
                    related_resources=["Cardiology", "Emergency Services"]
                ))
            
            elif category == SymptomCategory.NEUROLOGICAL:
                if high_severity:
                    recommendations.append(HealthRecommendation(
                        category="Neurological Health",
                        title="Neurological Symptoms Alert",
                        description="Severe neurological symptoms may indicate serious conditions "
                                  "requiring immediate evaluation.",
                        priority=SeverityLevel.HIGH,
                        actionable_steps=[
                            "Seek immediate medical attention for severe headache with stiff neck",
                            "Note any visual changes, weakness, or confusion",
                            "Keep a symptom diary with timing and triggers",
                            "Avoid driving until cleared by physician"
                        ],
                        follow_up_timeline="Immediate",
                        related_resources=["Neurology", "Emergency Services"]
                    ))
        
        return recommendations
    
    def _vital_signs_recommendations(self, vital_signs: VitalSigns, 
                                      patient: PatientProfile) -> List[HealthRecommendation]:
        """Generate recommendations based on vital signs."""
        recommendations = []
        
        # Blood pressure recommendations
        if vital_signs.blood_pressure_systolic > 140:
            recommendations.append(HealthRecommendation(
                category="Hypertension Management",
                title="Elevated Blood Pressure",
                description=f"Your blood pressure of {vital_signs.blood_pressure_systolic}/"
                          f"{vital_signs.blood_pressure_diastolic} mmHg is elevated. "
                          "Long-term hypertension management is crucial for preventing complications.",
                priority=SeverityLevel.HIGH,
                actionable_steps=[
                    "Monitor blood pressure twice daily",
                    "Reduce dietary sodium to <2300mg/day",
                    "Increase potassium-rich foods",
                    "Engage in moderate aerobic exercise",
                    "Limit alcohol to moderate levels",
                    "Practice stress reduction techniques",
                    "Discuss medication options with physician"
                ],
                follow_up_timeline="Within 1-2 weeks",
                related_resources=["DASH Diet", "Blood Pressure Monitoring Guide"]
            ))
        
        # BMI recommendations
        if vital_signs.bmi > 30:
            recommendations.append(HealthRecommendation(
                category="Weight Management",
                title="Obesity Management Program",
                description=f"Your BMI of {vital_signs.bmi} indicates obesity, "
                          "which increases risk for multiple health conditions.",
                priority=SeverityLevel.MODERATE,
                actionable_steps=[
                    "Calculate daily caloric needs (TDEE)",
                    "Create moderate caloric deficit (500-750 kcal/day)",
                    "Begin with 150 min/week moderate exercise",
                    "Focus on whole, unprocessed foods",
                    "Consider behavioral counseling",
                    "Set realistic weight loss goals (5-10% over 6 months)"
                ],
                follow_up_timeline="Monthly progress checks",
                related_resources=["Registered Dietitian", "Weight Management Clinic"]
            ))
        
        # Blood glucose recommendations
        if vital_signs.blood_glucose and vital_signs.blood_glucose > 100:
            priority = SeverityLevel.HIGH if vital_signs.blood_glucose > 126 else SeverityLevel.MODERATE
            recommendations.append(HealthRecommendation(
                category="Glucose Management",
                title="Blood Glucose Control",
                description=f"Your blood glucose of {vital_signs.blood_glucose} mg/dL requires attention. "
                          "Proper glucose management prevents long-term complications.",
                priority=priority,
                actionable_steps=[
                    "Schedule HbA1c test for 3-month average",
                    "Monitor fasting and post-meal glucose",
                    "Follow consistent meal timing",
                    "Choose low glycemic index foods",
                    "Begin or increase physical activity",
                    "Lose 5-7% body weight if overweight"
                ],
                follow_up_timeline="Within 1 week",
                related_resources=["Diabetes Education", "Endocrinology"]
            ))
        
        return recommendations
    
    def _risk_based_recommendations(self, risk_assessments: List[RiskAssessment]) -> List[HealthRecommendation]:
        """Generate recommendations based on risk assessments."""
        recommendations = []
        
        for assessment in risk_assessments:
            if assessment.risk_level in [RiskCategory.HIGH, RiskCategory.VERY_HIGH]:
                recommendations.append(HealthRecommendation(
                    category=assessment.category,
                    title=f"{assessment.category} Risk Reduction",
                    description=f"Your {assessment.risk_level.value} for {assessment.category} "
                              f"(score: {assessment.score}/100) requires proactive management. "
                              f"Risk factors: {', '.join(assessment.factors[:3])}.",
                    priority=SeverityLevel.HIGH,
                    actionable_steps=assessment.mitigation_strategies,
                    follow_up_timeline="Within 1 month",
                    related_resources=[f"{assessment.category} Specialist", "Preventive Care"]
                ))
        
        return recommendations
    
    def _lifestyle_recommendations(self, patient: PatientProfile, 
                                    vital_signs: VitalSigns) -> List[HealthRecommendation]:
        """Generate lifestyle-based recommendations."""
        recommendations = []
        lifestyle = patient.lifestyle_factors
        
        # Exercise recommendation
        if lifestyle.get("sedentary", False) or not lifestyle.get("exercise_routinely", True):
            recommendations.append(HealthRecommendation(
                category="Physical Activity",
                title="Increase Physical Activity",
                description="Regular physical activity is essential for preventing chronic diseases "
                          "and maintaining overall health. Current guidelines recommend at least "
                          "150 minutes of moderate-intensity exercise per week.",
                priority=SeverityLevel.MODERATE,
                actionable_steps=[
                    "Start with 10-15 minute walks daily",
                    "Gradually increase to 30 minutes most days",
                    "Include both aerobic and strength exercises",
                    "Choose activities you enjoy",
                    "Set specific, measurable activity goals",
                    "Consider a fitness tracker for motivation"
                ],
                follow_up_timeline="Ongoing with monthly progress review",
                related_resources=["Exercise Physiology", "Fitness Programs"]
            ))
        
        # Sleep recommendation
        if lifestyle.get("poor_sleep", False):
            recommendations.append(HealthRecommendation(
                category="Sleep Health",
                title="Improve Sleep Quality",
                description="Quality sleep is fundamental to health, affecting mental clarity, "
                          "immune function, and chronic disease risk. Adults need 7-9 hours "
                          "of quality sleep per night.",
                priority=SeverityLevel.MODERATE,
                actionable_steps=[
                    "Maintain consistent sleep schedule",
                    "Create a dark, cool sleep environment",
                    "Avoid screens 1 hour before bed",
                    "Limit caffeine after 2 PM",
                    "Avoid large meals close to bedtime",
                    "Consider relaxation techniques before sleep"
                ],
                follow_up_timeline="2-4 weeks to establish pattern",
                related_resources=["Sleep Hygiene Guide", "Sleep Study (if needed)"]
            ))
        
        # Smoking cessation
        if lifestyle.get("smoking", False):
            recommendations.append(HealthRecommendation(
                category="Smoking Cessation",
                title="Quit Smoking Program",
                description="Smoking is a leading cause of preventable disease and death. "
                          "Quitting smoking provides immediate and long-term health benefits, "
                          "regardless of how long you've smoked.",
                priority=SeverityLevel.HIGH,
                actionable_steps=[
                    "Set a quit date within the next 2 weeks",
                    "Identify triggers and plan coping strategies",
                    "Consider nicotine replacement therapy",
                    "Join a support group or program",
                    "Download quit-smoking app for tracking",
                    "Discuss prescription options with physician"
                ],
                follow_up_timeline="Immediate start with ongoing support",
                related_resources=["Quit Line", "Smoking Cessation Programs"]
            ))
        
        return recommendations
    
    def _diagnosis_followup_recommendations(self, 
                                             diagnosis_results: List[DiagnosisResult]) -> List[HealthRecommendation]:
        """Generate follow-up recommendations based on diagnosis results."""
        recommendations = []
        
        for diagnosis in diagnosis_results[:3]:  # Top 3 diagnoses
            if diagnosis.confidence_score > 50 and diagnosis.urgency_level in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
                recommendations.append(HealthRecommendation(
                    category="Medical Follow-up",
                    title=f"Evaluation for {diagnosis.condition}",
                    description=f"Based on your symptoms, there is a {diagnosis.confidence_score:.0f}% "
                              f"match with {diagnosis.condition}. {diagnosis.description}",
                    priority=diagnosis.urgency_level,
                    actionable_steps=[
                        f"Schedule appointment for evaluation",
                        f"Requested tests: {', '.join(diagnosis.recommended_tests[:3])}",
                        "Document any changes in symptoms",
                        "Prepare questions for your healthcare provider"
                    ],
                    follow_up_timeline="Within 24-48 hours" if diagnosis.urgency_level == SeverityLevel.CRITICAL else "Within 1 week",
                    related_resources=["Primary Care", "Specialist Referral"]
                ))
        
        return recommendations
    
    def _deduplicate_recommendations(self, 
                                      recommendations: List[HealthRecommendation]) -> List[HealthRecommendation]:
        """Remove duplicate recommendations based on title."""
        seen_titles = set()
        unique = []
        for rec in recommendations:
            if rec.title not in seen_titles:
                seen_titles.add(rec.title)
                unique.append(rec)
        return unique


# =============================================================================
# HEALTH REPORT GENERATOR
# =============================================================================

class HealthReportGenerator:
    """Generates comprehensive health reports."""
    
    def __init__(self):
        pass
    
    def generate_full_report(self, patient: PatientProfile,
                              vital_signs: VitalSigns,
                              symptoms: List[Symptom],
                              diagnosis_results: List[DiagnosisResult],
                              risk_assessments: List[RiskAssessment],
                              recommendations: List[HealthRecommendation],
                              vital_signs_analysis: Dict) -> Dict[str, Any]:
        """Generate a comprehensive health report."""
        report = {
            "report_metadata": {
                "report_id": self._generate_report_id(),
                "generated_at": datetime.datetime.now().isoformat(),
                "report_type": "Comprehensive Health Assessment",
                "version": "1.0"
            },
            "patient_summary": {
                **patient.to_dict(),
                "vital_signs_summary": vital_signs.to_dict()
            },
            "symptom_analysis": {
                "total_symptoms": len(symptoms),
                "symptoms_by_category": self._group_symptoms_by_category(symptoms),
                "symptom_details": [s.to_dict() for s in symptoms]
            },
            "vital_signs_analysis": vital_signs_analysis,
            "diagnosis_results": {
                "total_potential_conditions": len(diagnosis_results),
                "top_conditions": [d.to_dict() for d in diagnosis_results[:5]],
                "urgent_conditions": [d.to_dict() for d in diagnosis_results 
                                     if d.urgency_level in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]]
            },
            "risk_assessment": {
                "overall_risk_summary": self._summarize_risks(risk_assessments),
                "detailed_assessments": [r.to_dict() for r in risk_assessments]
            },
            "recommendations": {
                "total_recommendations": len(recommendations),
                "by_priority": self._group_recommendations_by_priority(recommendations),
                "detailed_recommendations": [r.to_dict() for r in recommendations]
            },
            "action_plan": self._create_action_plan(diagnosis_results, risk_assessments, recommendations),
            "next_steps": self._generate_next_steps(diagnosis_results, recommendations),
            "disclaimer": "This report is generated by MedVision AI System for informational purposes only. "
                         "Always consult qualified healthcare professionals for medical decisions."
        }
        
        return report
    
    def _generate_report_id(self) -> str:
        """Generate a unique report ID."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = random.randint(1000, 9999)
        return f"MVR-{timestamp}-{random_suffix}"
    
    def _group_symptoms_by_category(self, symptoms: List[Symptom]) -> Dict[str, int]:
        """Group symptoms by category."""
        grouped = {}
        for symptom in symptoms:
            category = symptom.category.value
            grouped[category] = grouped.get(category, 0) + 1
        return grouped
    
    def _summarize_risks(self, risk_assessments: List[RiskAssessment]) -> Dict[str, Any]:
        """Summarize risk assessments."""
        if not risk_assessments:
            return {"message": "No risk assessments available"}
        
        highest_risk = max(risk_assessments, key=lambda x: x.score)
        avg_score = statistics.mean([r.score for r in risk_assessments])
        
        return {
            "highest_risk_category": highest_risk.category,
            "highest_risk_level": highest_risk.risk_level.value,
            "highest_risk_score": highest_risk.score,
            "average_risk_score": round(avg_score, 2),
            "categories_assessed": len(risk_assessments)
        }
    
    def _group_recommendations_by_priority(self, 
                                            recommendations: List[HealthRecommendation]) -> Dict[str, int]:
        """Group recommendations by priority."""
        grouped = {}
        for rec in recommendations:
            priority = rec.priority.value
            grouped[priority] = grouped.get(priority, 0) + 1
        return grouped
    
    def _create_action_plan(self, diagnosis_results: List[DiagnosisResult],
                             risk_assessments: List[RiskAssessment],
                             recommendations: List[HealthRecommendation]) -> List[Dict]:
        """Create a prioritized action plan."""
        actions = []
        
        # Immediate actions (critical/urgent)
        critical_diagnoses = [d for d in diagnosis_results 
                             if d.urgency_level == SeverityLevel.CRITICAL]
        critical_recs = [r for r in recommendations if r.priority == SeverityLevel.CRITICAL]
        
        if critical_diagnoses or critical_recs:
            actions.append({
                "priority": "Immediate",
                "timeline": "Within 24 hours",
                "actions": ["Seek immediate medical attention", "Contact healthcare provider"]
            })
        
        # High priority actions
        high_diagnoses = [d for d in diagnosis_results 
                         if d.urgency_level == SeverityLevel.HIGH]
        high_recs = [r for r in recommendations if r.priority == SeverityLevel.HIGH]
        
        if high_diagnoses or high_recs:
            action_list = []
            for d in high_diagnoses[:2]:
                action_list.append(f"Schedule evaluation for {d.condition}")
            for r in high_recs[:3]:
                action_list.append(r.title)
            
            actions.append({
                "priority": "High",
                "timeline": "Within 1 week",
                "actions": action_list
            })
        
        # Moderate priority actions
        moderate_recs = [r for r in recommendations if r.priority == SeverityLevel.MODERATE]
        if moderate_recs:
            actions.append({
                "priority": "Moderate",
                "timeline": "Within 1 month",
                "actions": [r.title for r in moderate_recs[:5]]
            })
        
        # Long-term actions
        low_recs = [r for r in recommendations if r.priority == SeverityLevel.LOW]
        if low_recs:
            actions.append({
                "priority": "Long-term",
                "timeline": "Ongoing",
                "actions": [r.title for r in low_recs[:3]]
            })
        
        return actions
    
    def _generate_next_steps(self, diagnosis_results: List[DiagnosisResult],
                              recommendations: List[HealthRecommendation]) -> List[str]:
        """Generate clear next steps for the patient."""
        steps = []
        
        # Add diagnosis-based next steps
        if diagnosis_results:
            top_diagnosis = diagnosis_results[0]
            if top_diagnosis.confidence_score > 50:
                steps.append(f"Schedule medical evaluation for suspected {top_diagnosis.condition}")
        
        # Add recommendation-based next steps
        high_priority_recs = [r for r in recommendations if r.priority in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]]
        for rec in high_priority_recs[:2]:
            if rec.actionable_steps:
                steps.append(rec.actionable_steps[0])
        
        # General next steps
        steps.extend([
            "Keep a health journal to track symptoms",
            "Prepare questions for your next medical appointment",
            "Follow up on recommended diagnostic tests"
        ])
        
        return steps[:5]  # Limit to top 5


# =============================================================================
# MAIN MEDVISION SYSTEM
# =============================================================================

class MedVisionSystem:
    """
    Main MedVision AI Smart Health Analysis and Recommendation System.
    
    This class orchestrates all components to provide comprehensive health analysis:
    - Symptom analysis
    - Vital signs evaluation
    - Risk assessment
    - AI-powered recommendations
    - Health report generation
    """
    
    def __init__(self):
        """Initialize the MedVision system with all components."""
        self.knowledge_base = MedicalKnowledgeBase()
        self.symptom_analyzer = SymptomAnalyzer(self.knowledge_base)
        self.vital_signs_analyzer = VitalSignsAnalyzer(self.knowledge_base)
        self.risk_engine = RiskAssessmentEngine(self.knowledge_base)
        self.recommendation_engine = AIRecommendationEngine(self.knowledge_base)
        self.report_generator = HealthReportGenerator()
        
        self.analysis_history = []
        print("╔════════════════════════════════════════════════════════════════╗")
        print("║      MedVision – AI Smart Health Analysis System v1.0         ║")
        print("║           Successfully Initialized                             ║")
        print("╚════════════════════════════════════════════════════════════════╝")
    
    def analyze_health(self, patient: PatientProfile,
                       vital_signs: VitalSigns,
                       symptoms: List[Symptom]) -> Dict[str, Any]:
        """
        Perform comprehensive health analysis.
        
        Args:
            patient: Patient profile with demographics and history
            vital_signs: Current vital signs measurements
            symptoms: List of reported symptoms
        
        Returns:
            Comprehensive analysis report
        """
        print(f"\n{'='*60}")
        print(f"Analyzing health for {patient.name} (ID: {patient.patient_id})...")
        print(f"{'='*60}")
        
        # Step 1: Analyze symptoms
        print("\n[1/5] Analyzing symptoms...")
        diagnosis_results = self.symptom_analyzer.analyze_symptoms(symptoms)
        print(f"   → Found {len(diagnosis_results)} potential conditions")
        
        # Step 2: Analyze vital signs
        print("\n[2/5] Evaluating vital signs...")
        vital_signs_analysis = self.vital_signs_analyzer.analyze_vital_signs(vital_signs, patient)
        print(f"   → Overall status: {vital_signs_analysis['overall_status']}")
        
        # Step 3: Risk assessment
        print("\n[3/5] Assessing health risks...")
        risk_assessments = self.risk_engine.comprehensive_risk_assessment(patient, vital_signs)
        print(f"   → Assessed {len(risk_assessments)} risk categories")
        
        # Step 4: Generate recommendations
        print("\n[4/5] Generating AI recommendations...")
        recommendations = self.recommendation_engine.generate_recommendations(
            patient, vital_signs, symptoms, risk_assessments, diagnosis_results
        )
        print(f"   → Generated {len(recommendations)} recommendations")
        
        # Step 5: Generate report
        print("\n[5/5] Creating comprehensive report...")
        report = self.report_generator.generate_full_report(
            patient, vital_signs, symptoms, diagnosis_results,
            risk_assessments, recommendations, vital_signs_analysis
        )
        print(f"   → Report ID: {report['report_metadata']['report_id']}")
        
        # Store in history
        self.analysis_history.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "patient_id": patient.patient_id,
            "report_id": report['report_metadata']['report_id']
        })
        
        return report
    
    def quick_symptom_check(self, symptoms: List[str]) -> List[Dict]:
        """
        Quick symptom check without full patient profile.
        
        Args:
            symptoms: List of symptom names
        
        Returns:
            Potential conditions
        """
        symptom_objects = [
            Symptom(
                name=s,
                category=SymptomCategory.GENERAL,
                severity=SeverityLevel.MODERATE,
                duration_days=1
            ) for s in symptoms
        ]
        
        return [d.to_dict() for d in self.symptom_analyzer.analyze_symptoms(symptom_objects)]
    
    def get_symptom_suggestions(self, partial: str) -> List[str]:
        """Get symptom suggestions for autocomplete."""
        return self.symptom_analyzer.get_symptom_suggestions(partial)
    
    def get_all_known_symptoms(self) -> List[str]:
        """Get all symptoms in the knowledge base."""
        return self.knowledge_base.get_all_symptoms()
    
    def export_report(self, report: Dict[str, Any], filename: str = None) -> str:
        """
        Export report to JSON file.
        
        Args:
            report: Report dictionary
            filename: Optional filename
        
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"medvision_report_{report['report_metadata']['report_id']}.json"
        
        filepath = f"/home/z/my-project/download/{filename}"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return filepath


# =============================================================================
# CLI INTERFACE
# =============================================================================

def display_header():
    """Display application header."""
    print("\n" + "="*70)
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                                                                    ║")
    print("║   ███╗   ███╗███████╗ █████╗ ██╗███╗   ██╗███████╗                ║")
    print("║   ████╗ ████║██╔════╝██╔══██╗██║████╗  ██║██╔════╝                ║")
    print("║   ██╔████╔██║█████╗  ███████║██║██╔██╗ ██║█████╗                  ║")
    print("║   ██║╚██╔╝██║██╔══╝  ██╔══██║██║██║╚██╗██║██╔══╝                  ║")
    print("║   ██║ ╚═╝ ██║███████╗██║  ██║██║██║ ╚████║███████╗                ║")
    print("║   ╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝                ║")
    print("║                                                                    ║")
    print("║        AI Smart Health Analysis and Recommendation System          ║")
    print("║                          Version 1.0.0                             ║")
    print("║                                                                    ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print("="*70)
    print("\n⚠️  DISCLAIMER: This system is for educational purposes only.")
    print("    Always consult qualified healthcare professionals for medical advice.")
    print("="*70 + "\n")


def get_patient_input() -> PatientProfile:
    """Get patient information from user input."""
    print("\n" + "-"*50)
    print("PATIENT INFORMATION")
    print("-"*50)
    
    patient_id = input("Patient ID: ").strip() or f"P{random.randint(10000, 99999)}"
    name = input("Full Name: ").strip() or "Anonymous"
    
    while True:
        try:
            age = int(input("Age: "))
            break
        except ValueError:
            print("Please enter a valid age.")
    
    gender = input("Gender (M/F/Other): ").strip().upper() or "Not Specified"
    blood_type = input("Blood Type (A/B/AB/O+/−): ").strip().upper() or "Unknown"
    
    # Medical history
    print("\nMedical History (comma-separated, press Enter to skip):")
    history_input = input("> ").strip()
    medical_history = [h.strip() for h in history_input.split(",") if h.strip()]
    
    # Allergies
    print("\nAllergies (comma-separated, press Enter to skip):")
    allergies_input = input("> ").strip()
    allergies = [a.strip() for a in allergies_input.split(",") if a.strip()]
    
    # Current medications
    print("\nCurrent Medications (comma-separated, press Enter to skip):")
    meds_input = input("> ").strip()
    current_medications = [m.strip() for m in meds_input.split(",") if m.strip()]
    
    # Lifestyle factors
    lifestyle = {}
    print("\nLifestyle Factors:")
    lifestyle["smoking"] = input("  Smoking (Y/N): ").strip().upper() == "Y"
    lifestyle["sedentary"] = input("  Sedentary lifestyle (Y/N): ").strip().upper() == "Y"
    lifestyle["exercise_routinely"] = input("  Exercise regularly (Y/N): ").strip().upper() == "Y"
    lifestyle["family_history_cv"] = input("  Family history of heart disease (Y/N): ").strip().upper() == "Y"
    lifestyle["family_history_diabetes"] = input("  Family history of diabetes (Y/N): ").strip().upper() == "Y"
    
    return PatientProfile(
        patient_id=patient_id,
        name=name,
        age=age,
        gender=gender,
        blood_type=blood_type,
        medical_history=medical_history,
        allergies=allergies,
        current_medications=current_medications,
        lifestyle_factors=lifestyle
    )


def get_vital_signs_input() -> VitalSigns:
    """Get vital signs from user input."""
    print("\n" + "-"*50)
    print("VITAL SIGNS")
    print("-"*50)
    
    def get_float(prompt: str) -> float:
        while True:
            try:
                return float(input(prompt))
            except ValueError:
                print("Please enter a valid number.")
    
    def get_int(prompt: str) -> int:
        while True:
            try:
                return int(input(prompt))
            except ValueError:
                print("Please enter a valid number.")
    
    heart_rate = get_int("Heart Rate (bpm): ")
    
    print("Blood Pressure:")
    bp_systolic = get_int("  Systolic (mmHg): ")
    bp_diastolic = get_int("  Diastolic (mmHg): ")
    
    temperature = get_float("Temperature (°C): ")
    respiratory_rate = get_int("Respiratory Rate (breaths/min): ")
    oxygen_saturation = get_float("Oxygen Saturation (%): ")
    weight = get_float("Weight (kg): ")
    height = get_float("Height (cm): ")
    
    # Optional values
    print("\nOptional Measurements (press Enter to skip):")
    glucose_input = input("Blood Glucose (mg/dL): ").strip()
    blood_glucose = float(glucose_input) if glucose_input else None
    
    cholesterol_input = input("Total Cholesterol (mg/dL): ").strip()
    cholesterol = float(cholesterol_input) if cholesterol_input else None
    
    return VitalSigns(
        heart_rate=heart_rate,
        blood_pressure_systolic=bp_systolic,
        blood_pressure_diastolic=bp_diastolic,
        temperature=temperature,
        respiratory_rate=respiratory_rate,
        oxygen_saturation=oxygen_saturation,
        weight=weight,
        height=height,
        blood_glucose=blood_glucose,
        cholesterol_total=cholesterol
    )


def get_symptoms_input(system: MedVisionSystem) -> List[Symptom]:
    """Get symptoms from user input."""
    print("\n" + "-"*50)
    print("SYMPTOMS")
    print("-"*50)
    
    print("\nAvailable symptoms in knowledge base:")
    all_symptoms = system.get_all_known_symptoms()
    for i, symptom in enumerate(all_symptoms[:20]):
        print(f"  • {symptom}")
    if len(all_symptoms) > 20:
        print(f"  ... and {len(all_symptoms) - 20} more")
    
    print("\nEnter symptoms (type 'done' when finished):")
    
    symptoms = []
    symptom_count = 1
    
    while True:
        print(f"\nSymptom #{symptom_count}:")
        name = input("  Name (or 'done'): ").strip()
        
        if name.lower() == 'done':
            break
        
        if not name:
            continue
        
        # Get suggestions
        suggestions = system.get_symptom_suggestions(name)
        if suggestions and suggestions[0].lower() != name.lower():
            print(f"  Suggestions: {', '.join(suggestions[:5])}")
            use_suggestion = input("  Use suggestion? (number or Enter to keep): ").strip()
            if use_suggestion.isdigit() and int(use_suggestion) <= len(suggestions):
                name = suggestions[int(use_suggestion) - 1]
        
        # Severity
        print("  Severity: 1=Low, 2=Moderate, 3=High, 4=Critical")
        sev_input = input("  Select (1-4, default=2): ").strip()
        severity_map = {"1": SeverityLevel.LOW, "2": SeverityLevel.MODERATE,
                       "3": SeverityLevel.HIGH, "4": SeverityLevel.CRITICAL}
        severity = severity_map.get(sev_input, SeverityLevel.MODERATE)
        
        # Duration
        duration_input = input("  Duration (days, default=1): ").strip()
        duration = int(duration_input) if duration_input.isdigit() else 1
        
        # Category
        print("  Categories: 1=General, 2=Respiratory, 3=Cardiovascular, 4=Digestive,")
        print("              5=Neurological, 6=Musculoskeletal, 7=Psychological")
        cat_input = input("  Category (default=1): ").strip()
        category_map = {
            "1": SymptomCategory.GENERAL, "2": SymptomCategory.RESPIRATORY,
            "3": SymptomCategory.CARDIOVASCULAR, "4": SymptomCategory.DIGESTIVE,
            "5": SymptomCategory.NEUROLOGICAL, "6": SymptomCategory.MUSCULOSKELETAL,
            "7": SymptomCategory.PSYCHOLOGICAL
        }
        category = category_map.get(cat_input, SymptomCategory.GENERAL)
        
        symptom = Symptom(
            name=name,
            category=category,
            severity=severity,
            duration_days=duration
        )
        symptoms.append(symptom)
        symptom_count += 1
    
    return symptoms


def display_report_summary(report: Dict):
    """Display a summary of the analysis report."""
    print("\n" + "="*70)
    print("                        ANALYSIS SUMMARY")
    print("="*70)
    
    # Report metadata
    print(f"\n📋 Report ID: {report['report_metadata']['report_id']}")
    print(f"📅 Generated: {report['report_metadata']['generated_at']}")
    
    # Patient summary
    patient = report['patient_summary']
    print(f"\n👤 Patient: {patient['name']} ({patient['age']} years, {patient['gender']})")
    
    # Vital signs summary
    vs = patient['vital_signs_summary']
    print(f"\n📊 Vital Signs Summary:")
    print(f"   • Heart Rate: {vs['heart_rate']} bpm")
    print(f"   • Blood Pressure: {vs['blood_pressure_systolic']}/{vs['blood_pressure_diastolic']} mmHg ({vs['blood_pressure_category']})")
    print(f"   • Temperature: {vs['temperature']}°C")
    print(f"   • BMI: {vs['bmi']} ({vs['bmi_category']})")
    print(f"   • O₂ Saturation: {vs['oxygen_saturation']}%")
    
    # Symptoms
    symptoms = report['symptom_analysis']
    print(f"\n🤒 Symptoms Analyzed: {symptoms['total_symptoms']}")
    if symptoms['symptoms_by_category']:
        print("   By Category:")
        for cat, count in symptoms['symptoms_by_category'].items():
            print(f"     • {cat}: {count}")
    
    # Diagnosis results
    diagnoses = report['diagnosis_results']
    print(f"\n🔍 Potential Conditions: {diagnoses['total_potential_conditions']}")
    for i, diag in enumerate(diagnoses['top_conditions'][:3], 1):
        print(f"   {i}. {diag['condition']} (Confidence: {diag['confidence_score']:.1f}%)")
        print(f"      Matching: {', '.join(diag['matching_symptoms'][:3])}")
        print(f"      Urgency: {diag['urgency_level']}")
    
    # Risk assessment
    risks = report['risk_assessment']['overall_risk_summary']
    print(f"\n⚠️  Risk Assessment:")
    print(f"   • Highest Risk: {risks['highest_risk_category']} ({risks['highest_risk_level']})")
    print(f"   • Risk Score: {risks['highest_risk_score']}/100")
    
    # Recommendations
    recs = report['recommendations']
    print(f"\n💡 Recommendations: {recs['total_recommendations']}")
    for priority, count in recs['by_priority'].items():
        print(f"   • {priority} Priority: {count}")
    
    # Action plan
    print(f"\n📝 Action Plan:")
    for action in report['action_plan']:
        print(f"   [{action['priority']}] {action['timeline']}")
        for act in action['actions'][:3]:
            print(f"      → {act}")
    
    # Next steps
    print(f"\n🎯 Next Steps:")
    for i, step in enumerate(report['next_steps'], 1):
        print(f"   {i}. {step}")
    
    print("\n" + "="*70)


def interactive_demo():
    """Run an interactive demonstration of the system."""
    display_header()
    
    # Initialize system
    system = MedVisionSystem()
    
    while True:
        print("\n" + "-"*50)
        print("MAIN MENU")
        print("-"*50)
        print("1. Full Health Analysis")
        print("2. Quick Symptom Check")
        print("3. View All Known Symptoms")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            # Full analysis
            patient = get_patient_input()
            vital_signs = get_vital_signs_input()
            symptoms = get_symptoms_input(system)
            
            if not symptoms:
                print("\n⚠️  No symptoms entered. Using default symptom analysis.")
            
            # Perform analysis
            report = system.analyze_health(patient, vital_signs, symptoms)
            
            # Display summary
            display_report_summary(report)
            
            # Offer to export
            export = input("\nExport report to JSON? (Y/N): ").strip().upper()
            if export == "Y":
                filepath = system.export_report(report)
                print(f"✅ Report exported to: {filepath}")
        
        elif choice == "2":
            # Quick symptom check
            print("\nEnter symptoms (comma-separated):")
            symptoms_input = input("> ").strip()
            symptoms = [s.strip() for s in symptoms_input.split(",") if s.strip()]
            
            if symptoms:
                results = system.quick_symptom_check(symptoms)
                print(f"\n🔍 Found {len(results)} potential conditions:")
                for i, result in enumerate(results[:5], 1):
                    print(f"\n{i}. {result['condition']}")
                    print(f"   Confidence: {result['confidence_score']:.1f}%")
                    print(f"   Urgency: {result['urgency_level']}")
                    print(f"   Matching: {', '.join(result['matching_symptoms'][:3])}")
        
        elif choice == "3":
            # View all symptoms
            print("\n📚 All Known Symptoms:")
            symptoms = system.get_all_known_symptoms()
            for i, symptom in enumerate(symptoms, 1):
                print(f"  {i:2d}. {symptom}")
        
        elif choice == "4":
            print("\n👋 Thank you for using MedVision!")
            print("   Remember to consult healthcare professionals for medical advice.")
            break
        
        else:
            print("\n⚠️  Invalid option. Please select 1-4.")


def demo_mode():
    """Run a demonstration with sample data."""
    display_header()
    print("\n🎬 Running Demo Mode with Sample Data...\n")
    
    # Initialize system
    system = MedVisionSystem()
    
    # Create sample patient
    patient = PatientProfile(
        patient_id="DEMO-001",
        name="John Smith",
        age=45,
        gender="M",
        blood_type="A+",
        medical_history=["Hypertension"],
        allergies=["Penicillin"],
        current_medications=["Lisinopril 10mg"],
        lifestyle_factors={
            "smoking": False,
            "sedentary": True,
            "exercise_routinely": False,
            "family_history_cv": True,
            "family_history_diabetes": False
        }
    )
    
    # Create sample vital signs
    vital_signs = VitalSigns(
        heart_rate=82,
        blood_pressure_systolic=145,
        blood_pressure_diastolic=92,
        temperature=36.8,
        respiratory_rate=16,
        oxygen_saturation=97,
        weight=85,
        height=175,
        blood_glucose=110,
        cholesterol_total=225
    )
    
    # Create sample symptoms
    symptoms = [
        Symptom("headache", SymptomCategory.NEUROLOGICAL, SeverityLevel.MODERATE, 3),
        Symptom("fatigue", SymptomCategory.GENERAL, SeverityLevel.MODERATE, 7),
        Symptom("shortness of breath", SymptomCategory.RESPIRATORY, SeverityLevel.LOW, 5),
        Symptom("dizziness", SymptomCategory.NEUROLOGICAL, SeverityLevel.LOW, 2)
    ]
    
    # Perform analysis
    report = system.analyze_health(patient, vital_signs, symptoms)
    
    # Display summary
    display_report_summary(report)
    
    # Export report
    filepath = system.export_report(report)
    print(f"\n✅ Report exported to: {filepath}")
    
    return report


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        # Run demo mode
        demo_mode()
    else:
        # Run interactive mode
        interactive_demo()
