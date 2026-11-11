from pydantic import BaseModel, Field, constr

class PatientData(BaseModel):
    # string constraints to block dirty inputs (XSS/SQLi prevention)
    race: constr(strip_whitespace=True, max_length=50)
    gender: constr(strip_whitespace=True, max_length=20)
    age: constr(strip_whitespace=True, max_length=15)
    diag_1: constr(strip_whitespace=True, max_length=10)
    diag_2: constr(strip_whitespace=True, max_length=10)
    diag_3: constr(strip_whitespace=True, max_length=10)
    
    # numeric bounds based on clinical data
    time_in_hospital: int = Field(ge=1, le=14)
    num_lab_procedures: int = Field(ge=1)
    num_procedures: int = Field(ge=0)
    num_medications: int = Field(ge=1)
    number_outpatient: int = Field(ge=0)
    number_emergency: int = Field(ge=0)
    number_inpatient: int = Field(ge=0)
    number_diagnoses: int = Field(ge=1)
