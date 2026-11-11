from pydantic import BaseModel, Field, constr, ConfigDict

class PatientData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # Core Demographics & Admission
    race: constr(strip_whitespace=True, max_length=50) = "Caucasian"
    gender: constr(strip_whitespace=True, max_length=20) = "Female"
    age: constr(strip_whitespace=True, max_length=15) = "[70-80)"
    admission_type_id: int = 1
    discharge_disposition_id: int = 1
    admission_source_id: int = 7
    
    # Clinical Metrics
    time_in_hospital: int = Field(ge=1, le=14, default=5)
    num_lab_procedures: int = Field(ge=1, default=40)
    num_procedures: int = Field(ge=0, default=1)
    num_medications: int = Field(ge=1, default=15)
    number_outpatient: int = Field(ge=0, default=0)
    number_emergency: int = Field(ge=0, default=0)
    number_inpatient: int = Field(ge=0, default=1)
    number_diagnoses: int = Field(ge=1, default=9)
    
    # Diagnoses & Results
    diag_1: constr(strip_whitespace=True, max_length=10) = "250"
    diag_2: constr(strip_whitespace=True, max_length=10) = "401"
    diag_3: constr(strip_whitespace=True, max_length=10) = "428"
    max_glu_serum: constr(strip_whitespace=True, max_length=20) = "None"
    A1Cresult: constr(strip_whitespace=True, max_length=20) = "None"
    
    # Medications (Strict limits for XSS/SQLi prevention)
    metformin: constr(strip_whitespace=True, max_length=20) = "No"
    repaglinide: constr(strip_whitespace=True, max_length=20) = "No"
    nateglinide: constr(strip_whitespace=True, max_length=20) = "No"
    chlorpropamide: constr(strip_whitespace=True, max_length=20) = "No"
    glimepiride: constr(strip_whitespace=True, max_length=20) = "No"
    acetohexamide: constr(strip_whitespace=True, max_length=20) = "No"
    glipizide: constr(strip_whitespace=True, max_length=20) = "No"
    glyburide: constr(strip_whitespace=True, max_length=20) = "No"
    tolbutamide: constr(strip_whitespace=True, max_length=20) = "No"
    pioglitazone: constr(strip_whitespace=True, max_length=20) = "No"
    rosiglitazone: constr(strip_whitespace=True, max_length=20) = "No"
    acarbose: constr(strip_whitespace=True, max_length=20) = "No"
    miglitol: constr(strip_whitespace=True, max_length=20) = "No"
    troglitazone: constr(strip_whitespace=True, max_length=20) = "No"
    tolazamide: constr(strip_whitespace=True, max_length=20) = "No"
    examide: constr(strip_whitespace=True, max_length=20) = "No"
    citoglipton: constr(strip_whitespace=True, max_length=20) = "No"
    insulin: constr(strip_whitespace=True, max_length=20) = "No"
    
    # Hyphenated Medications mapped via Alias
    glyburide_metformin: constr(strip_whitespace=True, max_length=20) = Field(default="No", alias="glyburide-metformin")
    glipizide_metformin: constr(strip_whitespace=True, max_length=20) = Field(default="No", alias="glipizide-metformin")
    glimepiride_pioglitazone: constr(strip_whitespace=True, max_length=20) = Field(default="No", alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: constr(strip_whitespace=True, max_length=20) = Field(default="No", alias="metformin-rosiglitazone")
    metformin_pioglitazone: constr(strip_whitespace=True, max_length=20) = Field(default="No", alias="metformin-pioglitazone")
    
    change: constr(strip_whitespace=True, max_length=10) = "No"
    diabetesMed: constr(strip_whitespace=True, max_length=10) = "Yes"
